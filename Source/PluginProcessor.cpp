/*
  ==============================================================================
    PhaseCorrector - Plugin Processor Implementation
    4x Overlap-Add FFT processing with cubic spline phase interpolation
  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>
#include <algorithm>
#include <limits>

#ifndef JucePlugin_Name
    #define JucePlugin_Name "PhaseCorrector"
#endif

//==============================================================================
// Compensated Arithmetic for High-Precision Complex Multiplication
// Uses FMA (fused multiply-add) to eliminate catastrophic cancellation in ac-bd
//==============================================================================
namespace CompensatedArithmetic
{
    // Compute a*b - c*d with high accuracy using FMA
    // This eliminates catastrophic cancellation when a*b ≈ c*d
    inline double accurateDifferenceOfProducts(double a, double b, double c, double d)
    {
        double cd = c * d;
        double err = std::fma(c, d, -cd);  // Exact error: c*d - cd
        double result = std::fma(a, b, -cd);  // a*b - cd
        return result - err;  // (a*b - cd) - (c*d - cd) = a*b - c*d
    }

    // Compute a*b + c*d with high accuracy using FMA
    inline double accurateSumOfProducts(double a, double b, double c, double d)
    {
        double cd = c * d;
        double err = std::fma(c, d, -cd);  // Exact error: c*d - cd
        double result = std::fma(a, b, cd);  // a*b + cd
        return result + err;  // (a*b + cd) + (c*d - cd) = a*b + c*d
    }
}

//==============================================================================
// Cubic Spline Implementation (with crash protection)
//==============================================================================
void CubicSpline::clear()
{
    xData.clear();
    yData.clear();
    a.clear();
    b.clear();
    c.clear();
    d.clear();
    valid = false;
}

bool CubicSpline::build(const std::vector<double>& x, const std::vector<double>& y)
{
    clear();

    // Validate input
    if (x.size() != y.size() || x.size() < 2)
        return false;

    // Check for NaN/Inf values
    for (size_t i = 0; i < x.size(); ++i)
    {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
            return false;
    }

    // Check for strictly increasing x values
    for (size_t i = 1; i < x.size(); ++i)
    {
        if (x[i] <= x[i - 1])
            return false;
    }

    const size_t n = x.size();
    xData = x;
    yData = y;

    try
    {
        a.resize(n);
        b.resize(n - 1);
        c.resize(n);
        d.resize(n - 1);

        for (size_t i = 0; i < n; ++i)
            a[i] = y[i];

        std::vector<double> h(n - 1);
        for (size_t i = 0; i < n - 1; ++i)
        {
            h[i] = x[i + 1] - x[i];
            if (h[i] <= 0.0 || !std::isfinite(h[i]))
            {
                clear();
                return false;
            }
        }

        std::vector<double> alpha(n - 1, 0.0);
        for (size_t i = 1; i < n - 1; ++i)
        {
            double term1 = (3.0 / h[i]) * (a[i + 1] - a[i]);
            double term2 = (3.0 / h[i - 1]) * (a[i] - a[i - 1]);
            alpha[i] = term1 - term2;

            if (!std::isfinite(alpha[i]))
            {
                clear();
                return false;
            }
        }

        std::vector<double> l(n), mu(n), z(n);
        l[0] = 1.0;
        mu[0] = z[0] = 0.0;

        for (size_t i = 1; i < n - 1; ++i)
        {
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            if (std::abs(l[i]) < 1e-15)
            {
                clear();
                return false;
            }
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];

            if (!std::isfinite(mu[i]) || !std::isfinite(z[i]))
            {
                clear();
                return false;
            }
        }

        l[n - 1] = 1.0;
        z[n - 1] = c[n - 1] = 0.0;

        for (int j = static_cast<int>(n) - 2; j >= 0; --j)
        {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);

            if (!std::isfinite(b[j]) || !std::isfinite(c[j]) || !std::isfinite(d[j]))
            {
                clear();
                return false;
            }
        }

        valid = true;
        return true;
    }
    catch (...)
    {
        clear();
        return false;
    }
}

double CubicSpline::evaluate(double x) const
{
    if (!valid || xData.empty())
        return 0.0;

    // Clamp to range
    if (x <= xData.front())
        return yData.front();
    if (x >= xData.back())
        return yData.back();

    // Binary search
    auto it = std::upper_bound(xData.begin(), xData.end(), x);
    size_t i = static_cast<size_t>(std::distance(xData.begin(), it)) - 1;

    if (i >= b.size())
        i = b.size() - 1;

    double dx = x - xData[i];
    double result = a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;

    return std::isfinite(result) ? result : 0.0;
}

//==============================================================================
// Double-Precision FFT Implementation (Cooley-Tukey Radix-2)
//==============================================================================
DoubleFFT::DoubleFFT(int order)
{
    initialize(order);
}

void DoubleFFT::initialize(int order)
{
    fftOrder = order;
    fftSize = 1 << order;

    computeTwiddles();

    // Build bit-reversal table
    bitRevTable.resize(fftSize);
    for (int i = 0; i < fftSize; ++i)
    {
        int reversed = 0;
        int temp = i;
        for (int j = 0; j < fftOrder; ++j)
        {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }
        bitRevTable[i] = reversed;
    }

    workBuffer.resize(fftSize);
}

void DoubleFFT::computeTwiddles()
{
    twiddles.resize(fftSize / 2);
    const double twoPi = 2.0 * juce::MathConstants<double>::pi;

    for (int i = 0; i < fftSize / 2; ++i)
    {
        double angle = -twoPi * i / fftSize;
        twiddles[i] = std::complex<double>(std::cos(angle), std::sin(angle));
    }
}

void DoubleFFT::bitReverse(std::complex<double>* data)
{
    for (int i = 0; i < fftSize; ++i)
    {
        int j = bitRevTable[i];
        if (i < j)
            std::swap(data[i], data[j]);
    }
}

void DoubleFFT::fftCore(std::complex<double>* data, bool inverse)
{
    bitReverse(data);

    // Cooley-Tukey iterative FFT
    for (int stage = 1; stage <= fftOrder; ++stage)
    {
        int m = 1 << stage;
        int m2 = m >> 1;
        int twiddleStep = fftSize / m;

        for (int k = 0; k < fftSize; k += m)
        {
            for (int j = 0; j < m2; ++j)
            {
                std::complex<double> t = inverse
                    ? std::conj(twiddles[j * twiddleStep]) * data[k + j + m2]
                    : twiddles[j * twiddleStep] * data[k + j + m2];

                std::complex<double> u = data[k + j];
                data[k + j] = u + t;
                data[k + j + m2] = u - t;
            }
        }
    }

    // Scale for inverse transform
    if (inverse)
    {
        double scale = 1.0 / fftSize;
        for (int i = 0; i < fftSize; ++i)
            data[i] *= scale;
    }
}

void DoubleFFT::performRealForward(double* data)
{
    // Pack real data into complex array
    for (int i = 0; i < fftSize; ++i)
        workBuffer[i] = std::complex<double>(data[i], 0.0);

    fftCore(workBuffer.data(), false);

    // Unpack to interleaved real/imaginary format
    // data[0] = DC real, data[1] = DC imag (always 0 for real input)
    // data[2*k] = real[k], data[2*k+1] = imag[k]
    for (int i = 0; i <= fftSize / 2; ++i)
    {
        data[i * 2] = workBuffer[i].real();
        data[i * 2 + 1] = workBuffer[i].imag();
    }
}

void DoubleFFT::performRealInverse(double* data)
{
    // Pack interleaved complex data
    // Only positive frequencies are stored; reconstruct negative by conjugate symmetry
    workBuffer[0] = std::complex<double>(data[0], data[1]);

    for (int i = 1; i < fftSize / 2; ++i)
    {
        workBuffer[i] = std::complex<double>(data[i * 2], data[i * 2 + 1]);
        workBuffer[fftSize - i] = std::conj(workBuffer[i]);
    }

    workBuffer[fftSize / 2] = std::complex<double>(data[fftSize], data[fftSize + 1]);

    fftCore(workBuffer.data(), true);

    // Extract real part
    for (int i = 0; i < fftSize; ++i)
        data[i] = workBuffer[i].real();
}

//==============================================================================
// Phase Processor Implementation
//==============================================================================
PhaseProcessor::PhaseProcessor()
{
    // Initialize with default quality (High = 8192 FFT)
    reconfigure();
}

juce::StringArray PhaseProcessor::getQualityNames()
{
    return { "Low (1024)", "Medium (2048)", "High (4096)", "Very High (8192)", "Extreme (32k)",
             "Ultra (64k)", "Ultra (128k)", "Ultra (256k)" };
}

juce::StringArray PhaseProcessor::getOverlapNames()
{
    return { "50% (2x)", "75% (4x)", "87.5% (8x)", "93.75% (16x)", "96.875% (32x)", "98.4% (64x)" };
}

void PhaseProcessor::setQuality(Quality q)
{
    if (q != currentQuality)
    {
        currentQuality = q;
        needsReconfigure.store(true);
    }
}

void PhaseProcessor::setOverlap(Overlap o)
{
    if (o != currentOverlap)
    {
        currentOverlap = o;
        needsReconfigure.store(true);
    }
}

void PhaseProcessor::buildWindows()
{
    analysisWindow.resize(analysisSize);
    synthesisWindow.resize(analysisSize);

    // Use Hann window for all overlap modes - COLA compliant
    // Use double precision for window calculation
    // Use periodic Hann window (N in denominator) for proper COLA
    for (int i = 0; i < analysisSize; ++i)
    {
        double windowValue = 0.5 * (1.0 - std::cos(2.0 * juce::MathConstants<double>::pi * i / analysisSize));
        analysisWindow[i] = windowValue;
        synthesisWindow[i] = windowValue;
    }

    // Compute COLA compensation from window energy using Kahan summation
    // For overlap-add, compensation = hopSize / sum(window[i]²)
    // This formula gives consistent results for any FFT size
    double windowSquaredSum = 0.0;
    double windowSumError = 0.0;  // Kahan error compensation
    for (int i = 0; i < analysisSize; ++i)
    {
        const double value = analysisWindow[i] * synthesisWindow[i];
        const double y = value - windowSumError;
        const double t = windowSquaredSum + y;
        windowSumError = (t - windowSquaredSum) - y;
        windowSquaredSum = t;
    }

    if (windowSquaredSum > 0.001)
        windowCompensation = static_cast<double>(hopSize) / windowSquaredSum;
    else
        windowCompensation = 1.0;
}

void PhaseProcessor::reconfigure()
{
    juce::SpinLock::ScopedLockType lock(processingLock);

    // Calculate sizes based on quality
    // Analysis order determines the analysis window size
    int analysisOrder;
    switch (currentQuality)
    {
        case Quality::Low:       analysisOrder = 10; break;  // 1024 FFT
        case Quality::Medium:    analysisOrder = 11; break;  // 2048 FFT
        case Quality::High:      analysisOrder = 12; break;  // 4096 FFT
        case Quality::VeryHigh:  analysisOrder = 13; break;  // 8192 FFT
        case Quality::Extreme:   analysisOrder = 15; break;  // 32768 FFT
        case Quality::Ultra64k:  analysisOrder = 16; break;  // 65536 FFT
        case Quality::Ultra128k: analysisOrder = 17; break;  // 131072 FFT
        case Quality::Ultra256k: analysisOrder = 18; break;  // 262144 FFT
        default:                 analysisOrder = 12; break;
    }

    analysisSize = 1 << analysisOrder;
    fftSize = analysisSize;  // No zero-padding - cleaner for all-pass processing
    fftOrder = analysisOrder;

    // Calculate hop size based on overlap
    switch (currentOverlap)
    {
        case Overlap::Percent50:
            hopSize = analysisSize / 2;   // 50% overlap = 2x
            numOverlaps = 2;
            break;
        case Overlap::Percent75:
            hopSize = analysisSize / 4;   // 75% overlap = 4x
            numOverlaps = 4;
            break;
        case Overlap::Percent875:
            hopSize = analysisSize / 8;   // 87.5% overlap = 8x
            numOverlaps = 8;
            break;
        case Overlap::Percent9375:
            hopSize = analysisSize / 16;  // 93.75% overlap = 16x
            numOverlaps = 16;
            break;
        case Overlap::Percent96875:
            hopSize = analysisSize / 32;  // 96.875% overlap = 32x
            numOverlaps = 32;
            break;
        case Overlap::Percent984375:
            hopSize = analysisSize / 64;  // 98.4375% overlap = 64x
            numOverlaps = 64;
            break;
        default:
            hopSize = analysisSize / 16;
            numOverlaps = 16;
            break;
    }

    // Recreate double-precision FFT
    fft.initialize(fftOrder);

    // Rebuild windows
    buildWindows();

    // Resize buffers - need extra space for overlap-add
    int bufferSize = fftSize * 2;  // Extra space for safety
    for (auto& ch : channels)
        ch.resize(bufferSize, fftSize, hopSize);

    // Resize phase table (double precision)
    phaseTable.resize(fftSize / 2 + 1, 0.0);

    needsReconfigure.store(false);

    // Rebuild phase table with new size
    rebuildPhaseTable();
}

void PhaseProcessor::prepare(double sampleRate, int /*maxBlockSize*/)
{
    currentSampleRate = sampleRate;

    // Always reconfigure on prepare to ensure correct setup
    reconfigure();
}

void PhaseProcessor::reset()
{
    juce::SpinLock::ScopedLockType lock(processingLock);
    for (auto& ch : channels)
        ch.clear();
}

void PhaseProcessor::updatePhaseCurve(const std::vector<std::pair<double, double>>& points)
{
    std::lock_guard<std::mutex> lock(curveMutex);

    if (points.size() < 2)
    {
        phaseCurve.clear();
        std::fill(phaseTable.begin(), phaseTable.end(), 0.0);
        return;
    }

    // Sort and validate points
    auto sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Remove duplicates and invalid points
    std::vector<std::pair<double, double>> validPoints;
    validPoints.reserve(sortedPoints.size());

    double lastX = -std::numeric_limits<double>::infinity();
    for (const auto& p : sortedPoints)
    {
        if (std::isfinite(p.first) && std::isfinite(p.second) && p.first > lastX)
        {
            validPoints.push_back(p);
            lastX = p.first;
        }
    }

    if (validPoints.size() < 2)
    {
        phaseCurve.clear();
        std::fill(phaseTable.begin(), phaseTable.end(), 0.0);
        return;
    }

    std::vector<double> x, y;
    x.reserve(validPoints.size());
    y.reserve(validPoints.size());

    for (const auto& p : validPoints)
    {
        x.push_back(p.first);
        y.push_back(p.second);
    }

    if (!phaseCurve.build(x, y))
    {
        // Fallback to linear interpolation if spline fails
        phaseCurve.clear();
    }

    rebuildPhaseTable();
}

void PhaseProcessor::rebuildPhaseTable()
{
    const int numBins = fftSize / 2 + 1;
    const double depth = phaseDepth.load();

    // Ensure phase table is correct size
    if (static_cast<int>(phaseTable.size()) != numBins)
        phaseTable.resize(numBins, 0.0);

    // Also rebuild the impulse response for proper all-pass filtering
    rebuildImpulseResponse();

    for (int bin = 0; bin < numBins; ++bin)
    {
        double freq = static_cast<double>(bin) * currentSampleRate / static_cast<double>(fftSize);

        if (freq < MIN_FREQ || freq > MAX_FREQ || !phaseCurve.isValid())
        {
            phaseTable[bin] = 0.0;
            continue;
        }

        double logFreq = std::log10(std::max(freq, MIN_FREQ));
        double normalizedPhase = phaseCurve.evaluate(logFreq);

        // Clamp phase value
        normalizedPhase = juce::jlimit(-1.0, 1.0, normalizedPhase);

        // Convert to radians and apply depth (double precision)
        double phaseRadians = normalizedPhase * 2.0 * juce::MathConstants<double>::pi * depth;
        phaseTable[bin] = phaseRadians;
    }

    // Smooth transition at boundaries
    const int fadeLength = std::max(10, fftSize / 512);
    for (int i = 0; i < fadeLength; ++i)
    {
        double fade = static_cast<double>(i) / fadeLength;
        int lowBin = static_cast<int>((MIN_FREQ / currentSampleRate) * fftSize) + i;
        int highBin = static_cast<int>((MAX_FREQ / currentSampleRate) * fftSize) - i;

        if (lowBin >= 0 && lowBin < numBins)
            phaseTable[lowBin] *= fade;
        if (highBin >= 0 && highBin < numBins)
            phaseTable[highBin] *= fade;
    }
}

void PhaseProcessor::rebuildImpulseResponse()
{
    // Build the all-pass filter frequency response from the phase curve
    // H(k) = e^(j * phase(k)) where |H(k)| = 1 (all-pass)
    // Store as complex spectrum for fast convolution (double precision)

    const int numBins = fftSize / 2 + 1;
    const double depth = phaseDepth.load();

    // Resize filter spectrum buffer (interleaved real/imag for complex multiply)
    if (static_cast<int>(filterSpectrum.size()) != fftSize * 2)
        filterSpectrum.resize(fftSize * 2, 0.0);

    // Build complex frequency response: H(k) = e^(j * phase(k))
    for (int bin = 0; bin < numBins; ++bin)
    {
        double phase = 0.0;

        if (phaseCurve.isValid())
        {
            double freq = static_cast<double>(bin) * currentSampleRate / static_cast<double>(fftSize);

            if (freq >= MIN_FREQ && freq <= MAX_FREQ)
            {
                double logFreq = std::log10(std::max(freq, MIN_FREQ));
                double normalizedPhase = phaseCurve.evaluate(logFreq);
                normalizedPhase = juce::jlimit(-1.0, 1.0, normalizedPhase);
                phase = normalizedPhase * 2.0 * juce::MathConstants<double>::pi * depth;
            }
        }

        // H(k) = cos(phase) + j*sin(phase) (magnitude = 1, all-pass)
        filterSpectrum[bin * 2] = std::cos(phase);      // Real
        filterSpectrum[bin * 2 + 1] = std::sin(phase);  // Imaginary
    }

    filterIRReady.store(true);
}

void PhaseProcessor::processFrame(int channel)
{
    auto& ch = channels[channel];

    const int bufferSize = static_cast<int>(ch.inputBuffer.size());
    int readPos = ch.inputWritePos - analysisSize;
    if (readPos < 0)
        readPos += bufferSize;

    // Use pre-allocated FFT buffer (double precision)
    double* __restrict fftData = ch.fftBuffer.data();
    const double* __restrict inBuf = ch.inputBuffer.data();
    const double* __restrict anaWin = analysisWindow.data();

    // Clear FFT buffer
    std::memset(fftData, 0, fftSize * 2 * sizeof(double));

    // Apply analysis window
    if (readPos + analysisSize <= bufferSize)
    {
        for (int i = 0; i < analysisSize; ++i)
            fftData[i] = inBuf[readPos + i] * anaWin[i];
    }
    else
    {
        for (int i = 0; i < analysisSize; ++i)
        {
            int idx = (readPos + i) % bufferSize;
            fftData[i] = inBuf[idx] * anaWin[i];
        }
    }

    // Forward FFT (double precision)
    fft.performRealForward(fftData);

    // Apply all-pass filter via complex multiplication in frequency domain
    // This is proper convolution: Y(k) = X(k) * H(k)
    // Complex multiply: (a+jb)(c+jd) = (ac-bd) + j(ad+bc)
    const double depth = std::abs(phaseDepth.load());
    const bool hasPhaseData = phaseCurve.isValid() && depth > 0.001 && filterIRReady.load();

    if (hasPhaseData)
    {
        const double* __restrict filterSpec = filterSpectrum.data();
        const int numBins = fftSize / 2 + 1;

        for (int bin = 0; bin < numBins; ++bin)
        {
            const int idx = bin * 2;

            // Input spectrum (X)
            const double xReal = fftData[idx];
            const double xImag = fftData[idx + 1];

            // Filter spectrum (H) - all-pass: H(k) = e^(j*phase(k))
            const double hReal = filterSpec[idx];
            const double hImag = filterSpec[idx + 1];

            // Complex multiplication: Y = X * H (double precision)
            fftData[idx] = xReal * hReal - xImag * hImag;      // Real part
            fftData[idx + 1] = xReal * hImag + xImag * hReal;  // Imaginary part
        }
    }

    // Inverse FFT (double precision)
    fft.performRealInverse(fftData);

    // Overlap-add with synthesis window using Kahan compensated summation
    const int writePos = ch.outputReadPos;
    double* __restrict outBuf = ch.outputBuffer.data();
    double* __restrict outBufErr = ch.outputBufferError.data();
    const double* __restrict synWin = synthesisWindow.data();
    const double compensation = windowCompensation;

    if (writePos + analysisSize <= bufferSize)
    {
        for (int i = 0; i < analysisSize; ++i)
        {
            const int idx = writePos + i;
            const double value = fftData[i] * synWin[i] * compensation;
            // Kahan compensated summation for near-infinite precision accumulation
            const double y = value - outBufErr[idx];
            const double t = outBuf[idx] + y;
            outBufErr[idx] = (t - outBuf[idx]) - y;
            outBuf[idx] = t;
        }
    }
    else
    {
        for (int i = 0; i < analysisSize; ++i)
        {
            const int idx = (writePos + i) % bufferSize;
            const double value = fftData[i] * synWin[i] * compensation;
            // Kahan compensated summation for near-infinite precision accumulation
            const double y = value - outBufErr[idx];
            const double t = outBuf[idx] + y;
            outBufErr[idx] = (t - outBuf[idx]) - y;
            outBuf[idx] = t;
        }
    }
}

void PhaseProcessor::process(juce::AudioBuffer<float>& buffer)
{
    // Check if reconfiguration is needed (quality/overlap changed)
    if (needsReconfigure.load())
    {
        reconfigure();
    }

    juce::SpinLock::ScopedTryLockType lock(processingLock);
    if (!lock.isLocked())
        return;  // Skip processing if reconfiguring

    const int numChannels = std::min(buffer.getNumChannels(), 2);
    const int numSamples = buffer.getNumSamples();

    // Fast bypass: if no phase curve and depth is effectively zero, just pass through
    // with latency compensation (still need to maintain buffer state)
    const double depth = std::abs(phaseDepth.load());
    const bool shouldProcess = phaseCurve.isValid() && depth > 0.001;

    for (int sample = 0; sample < numSamples; ++sample)
    {
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto& state = channels[ch];
            const int bufferSize = static_cast<int>(state.inputBuffer.size());

            // Convert float input to double for internal processing
            state.inputBuffer[state.inputWritePos] = static_cast<double>(buffer.getSample(ch, sample));

            // Read output and convert double back to float
            double outputSample = state.outputBuffer[state.outputReadPos];
            state.outputBuffer[state.outputReadPos] = 0.0;
            state.outputBufferError[state.outputReadPos] = 0.0;  // Clear Kahan error

            buffer.setSample(ch, sample, static_cast<float>(outputSample));

            state.inputWritePos = (state.inputWritePos + 1) % bufferSize;
            state.outputReadPos = (state.outputReadPos + 1) % bufferSize;

            state.samplesUntilNextFrame--;
            if (state.samplesUntilNextFrame <= 0)
            {
                if (shouldProcess)
                {
                    processFrame(ch);
                }
                else
                {
                    // Bypass mode: just copy input to output with proper windowing
                    // to maintain correct latency behavior
                    processFrameBypass(ch);
                }
                state.samplesUntilNextFrame = hopSize;
            }
        }
    }
}

void PhaseProcessor::processFrameBypass(int channel)
{
    // Simplified frame processing for bypass mode - no FFT, just overlap-add (double precision)
    auto& ch = channels[channel];

    const int bufferSize = static_cast<int>(ch.inputBuffer.size());
    int readPos = ch.inputWritePos - analysisSize;
    if (readPos < 0)
        readPos += bufferSize;

    const int writePos = ch.outputReadPos;
    const double* __restrict inBuf = ch.inputBuffer.data();
    double* __restrict outBuf = ch.outputBuffer.data();
    double* __restrict outBufErr = ch.outputBufferError.data();
    const double* __restrict anaWin = analysisWindow.data();
    const double* __restrict synWin = synthesisWindow.data();
    const double compensation = windowCompensation;

    // Direct overlap-add without FFT using Kahan compensated summation
    for (int i = 0; i < analysisSize; ++i)
    {
        const int rIdx = (readPos + i) % bufferSize;
        const int wIdx = (writePos + i) % bufferSize;
        const double value = inBuf[rIdx] * anaWin[i] * synWin[i] * compensation;
        // Kahan compensated summation for near-infinite precision accumulation
        const double y = value - outBufErr[wIdx];
        const double t = outBuf[wIdx] + y;
        outBufErr[wIdx] = (t - outBuf[wIdx]) - y;
        outBuf[wIdx] = t;
    }
}

//==============================================================================
// 64-bit Native Double Precision Processing (no float conversion)
//==============================================================================
void PhaseProcessor::process(juce::AudioBuffer<double>& buffer)
{
    // Check if reconfiguration is needed (quality/overlap changed)
    if (needsReconfigure.load())
    {
        reconfigure();
    }

    juce::SpinLock::ScopedTryLockType lock(processingLock);
    if (!lock.isLocked())
        return;  // Skip processing if reconfiguring

    const int numChannels = std::min(buffer.getNumChannels(), 2);
    const int numSamples = buffer.getNumSamples();

    // Fast bypass: if no phase curve and depth is effectively zero, just pass through
    // with latency compensation (still need to maintain buffer state)
    const double depth = std::abs(phaseDepth.load());
    const bool shouldProcess = phaseCurve.isValid() && depth > 0.001;

    // Get raw pointers for direct access (more reliable than getSample/setSample)
    double* channelPtrs[2] = { nullptr, nullptr };
    for (int ch = 0; ch < numChannels; ++ch)
        channelPtrs[ch] = buffer.getWritePointer(ch);

    for (int sample = 0; sample < numSamples; ++sample)
    {
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto& state = channels[ch];
            const int bufferSize = static_cast<int>(state.inputBuffer.size());

            // Native double - no conversion needed!
            state.inputBuffer[state.inputWritePos] = channelPtrs[ch][sample];

            // Read output - native double, no conversion!
            double outputSample = state.outputBuffer[state.outputReadPos];
            state.outputBuffer[state.outputReadPos] = 0.0;
            state.outputBufferError[state.outputReadPos] = 0.0;  // Clear Kahan error

            channelPtrs[ch][sample] = outputSample;

            state.inputWritePos = (state.inputWritePos + 1) % bufferSize;
            state.outputReadPos = (state.outputReadPos + 1) % bufferSize;

            state.samplesUntilNextFrame--;
            if (state.samplesUntilNextFrame <= 0)
            {
                if (shouldProcess)
                {
                    processFrame(ch);
                }
                else
                {
                    // Bypass mode: just copy input to output with proper windowing
                    // to maintain correct latency behavior
                    processFrameBypass(ch);
                }
                state.samplesUntilNextFrame = hopSize;
            }
        }
    }
}

//==============================================================================
// CSV Parser Implementation (with robust error handling)
//==============================================================================
CSVParser::ParseResult CSVParser::parse(const juce::File& file)
{
    ParseResult result;

    if (!file.existsAsFile())
    {
        result.errorMessage = "File does not exist";
        return result;
    }

    juce::String content = file.loadFileAsString();
    if (content.isEmpty())
    {
        result.errorMessage = "File is empty";
        return result;
    }

    return parseString(content);
}

CSVParser::ParseResult CSVParser::parseString(const juce::String& content)
{
    ParseResult result;
    result.success = true;

    juce::StringArray lines;
    lines.addLines(content);

    result.points.reserve(static_cast<size_t>(lines.size()));

    for (const auto& line : lines)
    {
        auto trimmedLine = line.trim();
        if (trimmedLine.isEmpty())
            continue;

        // Skip comments
        if (trimmedLine.startsWithChar('#') || trimmedLine.startsWithChar('/') ||
            trimmedLine.startsWithChar(';'))
            continue;

        // Parse with multiple delimiter support
        juce::StringArray parts;
        if (trimmedLine.containsChar(';'))
            parts.addTokens(trimmedLine, ";", "\"");
        else if (trimmedLine.containsChar(','))
            parts.addTokens(trimmedLine, ",", "\"");
        else if (trimmedLine.containsChar('\t'))
            parts.addTokens(trimmedLine, "\t", "\"");
        else
            parts.addTokens(trimmedLine, " ", "\"");

        if (parts.size() < 2)
            continue;

        // Parse values with error checking
        double logFreq = parts[0].trim().getDoubleValue();
        double phaseValue = parts[1].trim().getDoubleValue();

        // Validate
        if (!std::isfinite(logFreq) || !std::isfinite(phaseValue))
            continue;

        // Accept slightly wider range for flexibility
        if (logFreq < LOG_MIN_FREQ - 0.5 || logFreq > LOG_MAX_FREQ + 0.5)
            continue;

        // Clamp phase
        phaseValue = juce::jlimit(-2.0, 2.0, phaseValue);

        result.points.emplace_back(logFreq, phaseValue);
    }

    if (result.points.empty())
    {
        result.success = false;
        result.errorMessage = "No valid data points found";
        return result;
    }

    // Sort by frequency
    std::sort(result.points.begin(), result.points.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Remove exact duplicates
    auto last = std::unique(result.points.begin(), result.points.end(),
                            [](const auto& a, const auto& b) { return std::abs(a.first - b.first) < 1e-9; });
    result.points.erase(last, result.points.end());

    result.pointCount = static_cast<int>(result.points.size());
    return result;
}

//==============================================================================
// Preset Manager Implementation
//==============================================================================
PresetManager::PresetManager(juce::AudioProcessorValueTreeState& apvtsRef,
                             std::function<std::vector<std::pair<double, double>>()> getCurveFunc,
                             std::function<void(const std::vector<std::pair<double, double>>&)> setCurveFunc)
    : apvts(apvtsRef), getCurve(getCurveFunc), setCurve(setCurveFunc)
{
}

juce::File PresetManager::getPresetDirectory() const
{
    auto dir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                   .getChildFile("PhaseCorrector")
                   .getChildFile("Presets");
    if (!dir.exists())
        dir.createDirectory();
    return dir;
}

juce::File PresetManager::getPresetFile(const juce::String& name) const
{
    return getPresetDirectory().getChildFile(name + ".xml");
}

void PresetManager::savePreset(const juce::String& name)
{
    if (name.isEmpty())
        return;

    auto state = apvts.copyState();

    // Save curve data
    auto curvePoints = getCurve();
    if (!curvePoints.empty())
    {
        juce::String curveData;
        for (const auto& point : curvePoints)
            curveData += juce::String(point.first, 6) + ";" + juce::String(point.second, 6) + "\n";
        state.setProperty("curveData", curveData, nullptr);
    }

    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    if (xml)
    {
        auto file = getPresetFile(name);
        xml->writeTo(file);
        currentPresetName = name;
    }
}

void PresetManager::loadPreset(const juce::String& name)
{
    auto file = getPresetFile(name);
    if (!file.existsAsFile())
        return;

    std::unique_ptr<juce::XmlElement> xml = juce::XmlDocument::parse(file);
    if (xml && xml->hasTagName(apvts.state.getType()))
    {
        auto state = juce::ValueTree::fromXml(*xml);
        apvts.replaceState(state);

        // Load curve data
        juce::String curveData = state.getProperty("curveData", "");
        if (curveData.isNotEmpty())
        {
            auto result = CSVParser::parseString(curveData);
            if (result.success)
                setCurve(result.points);
        }
        else
        {
            setCurve({});  // Clear curve if no data
        }

        currentPresetName = name;
    }
}

void PresetManager::deletePreset(const juce::String& name)
{
    auto file = getPresetFile(name);
    if (file.existsAsFile())
        file.deleteFile();

    if (currentPresetName == name)
        currentPresetName.clear();
}

juce::StringArray PresetManager::getPresetList() const
{
    juce::StringArray presets;
    auto dir = getPresetDirectory();

    for (const auto& file : dir.findChildFiles(juce::File::findFiles, false, "*.xml"))
        presets.add(file.getFileNameWithoutExtension());

    presets.sort(true);
    return presets;
}

//==============================================================================
// Plugin Processor Implementation
//==============================================================================
PhaseCorrectorAudioProcessor::PhaseCorrectorAudioProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout()),
      presetManager(apvts,
                    [this]() { return getCurrentCurvePoints(); },
                    [this](const std::vector<std::pair<double, double>>& pts) { setCurvePoints(pts); })
{
    apvts.addParameterListener("dryWet", this);
    apvts.addParameterListener("outputGain", this);
    apvts.addParameterListener("depth", this);
    apvts.addParameterListener("fftQuality", this);
    apvts.addParameterListener("fftOverlap", this);
}

PhaseCorrectorAudioProcessor::~PhaseCorrectorAudioProcessor()
{
    apvts.removeParameterListener("dryWet", this);
    apvts.removeParameterListener("outputGain", this);
    apvts.removeParameterListener("depth", this);
    apvts.removeParameterListener("fftQuality", this);
    apvts.removeParameterListener("fftOverlap", this);
}

juce::AudioProcessorValueTreeState::ParameterLayout PhaseCorrectorAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Dry/Wet
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("dryWet", 1), "Dry/Wet",
        juce::NormalisableRange<float>(0.0f, 100.0f, 0.1f), 100.0f,
        juce::AudioParameterFloatAttributes().withLabel("%")));

    // Output Gain
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("outputGain", 1), "Output Gain",
        juce::NormalisableRange<float>(-24.0f, 24.0f, 0.1f), 0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));

    // Phase Depth - negative values invert the phase curve (for correction)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("depth", 1), "Depth",
        juce::NormalisableRange<float>(-200.0f, 200.0f, 0.1f), 100.0f,
        juce::AudioParameterFloatAttributes().withLabel("%")));

    // FFT Quality - determines frequency resolution and latency
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID("fftQuality", 1), "FFT Quality",
        PhaseProcessor::getQualityNames(), 2));  // Default: High (8192)

    // Overlap amount - affects latency vs quality trade-off
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID("fftOverlap", 1), "Overlap",
        PhaseProcessor::getOverlapNames(), 5));  // Default: 98.4% (64x) for maximum quality

    return { params.begin(), params.end() };
}

void PhaseCorrectorAudioProcessor::parameterChanged(const juce::String& parameterID, float newValue)
{
    if (parameterID == "outputGain")
    {
        outputGainLinear.store(std::pow(10.0, static_cast<double>(newValue) / 20.0));
    }
    else if (parameterID == "depth")
    {
        phaseProcessor.setDepth(static_cast<double>(newValue) / 100.0);
    }
    else if (parameterID == "fftQuality")
    {
        auto newQuality = static_cast<PhaseProcessor::Quality>(static_cast<int>(newValue));
        phaseProcessor.setQuality(newQuality);
        // Update latency reporting immediately using expected latency for new quality
        setLatencySamples(PhaseProcessor::getLatencyForQuality(newQuality));
    }
    else if (parameterID == "fftOverlap")
    {
        phaseProcessor.setOverlap(static_cast<PhaseProcessor::Overlap>(static_cast<int>(newValue)));
    }
}

const juce::String PhaseCorrectorAudioProcessor::getName() const { return JucePlugin_Name; }
bool PhaseCorrectorAudioProcessor::acceptsMidi() const { return false; }
bool PhaseCorrectorAudioProcessor::producesMidi() const { return false; }
bool PhaseCorrectorAudioProcessor::isMidiEffect() const { return false; }
double PhaseCorrectorAudioProcessor::getTailLengthSeconds() const { return 0.1; }
int PhaseCorrectorAudioProcessor::getNumPrograms() { return 1; }
int PhaseCorrectorAudioProcessor::getCurrentProgram() { return 0; }
void PhaseCorrectorAudioProcessor::setCurrentProgram(int) {}
const juce::String PhaseCorrectorAudioProcessor::getProgramName(int) { return {}; }
void PhaseCorrectorAudioProcessor::changeProgramName(int, const juce::String&) {}

void PhaseCorrectorAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    lastSampleRate = sampleRate;
    lastBlockSize = samplesPerBlock;

    // Apply quality and overlap settings before preparing
    int quality = static_cast<int>(apvts.getRawParameterValue("fftQuality")->load());
    int overlap = static_cast<int>(apvts.getRawParameterValue("fftOverlap")->load());
    phaseProcessor.setQuality(static_cast<PhaseProcessor::Quality>(quality));
    phaseProcessor.setOverlap(static_cast<PhaseProcessor::Overlap>(overlap));

    // Prepare at native sample rate (no oversampling)
    phaseProcessor.prepare(sampleRate, samplesPerBlock);

    phaseProcessor.setDepth(static_cast<double>(apvts.getRawParameterValue("depth")->load()) / 100.0);

    // Initialize double-precision output gain
    outputGainLinear.store(std::pow(10.0, static_cast<double>(apvts.getRawParameterValue("outputGain")->load()) / 20.0));

    // Report latency to host for PDC
    setLatencySamples(phaseProcessor.getLatencySamples());
}

void PhaseCorrectorAudioProcessor::releaseResources()
{
    phaseProcessor.reset();
}

bool PhaseCorrectorAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    return layouts.getMainInputChannelSet() == layouts.getMainOutputChannelSet();
}

void PhaseCorrectorAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const int totalNumInputChannels = getTotalNumInputChannels();
    const int totalNumOutputChannels = getTotalNumOutputChannels();

    for (int i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    const double dryWet = static_cast<double>(apvts.getRawParameterValue("dryWet")->load()) / 100.0;

    // Store dry signal
    juce::AudioBuffer<float> dryBuffer;
    if (dryWet < 1.0)
        dryBuffer.makeCopyOf(buffer);

    // Process phase at native sample rate
    phaseProcessor.process(buffer);

    // Dry/Wet mix (double precision)
    if (dryWet < 1.0)
    {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            auto* wetData = buffer.getWritePointer(ch);
            const auto* dryData = dryBuffer.getReadPointer(ch);

            for (int i = 0; i < buffer.getNumSamples(); ++i)
            {
                wetData[i] = static_cast<float>(
                    static_cast<double>(dryData[i]) * (1.0 - dryWet) +
                    static_cast<double>(wetData[i]) * dryWet
                );
            }
        }
    }

    // Output gain (double precision)
    const double gain = outputGainLinear.load(std::memory_order_relaxed);
    if (gain != 1.0)
    {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            auto* data = buffer.getWritePointer(ch);
            for (int i = 0; i < buffer.getNumSamples(); ++i)
                data[i] = static_cast<float>(static_cast<double>(data[i]) * gain);
        }
    }
}

//==============================================================================
// 64-bit Native Double Precision Processing (VST3 hosts that support it)
// This eliminates the float conversion bottleneck entirely
//==============================================================================
void PhaseCorrectorAudioProcessor::processBlock(juce::AudioBuffer<double>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const int totalNumInputChannels = getTotalNumInputChannels();
    const int totalNumOutputChannels = getTotalNumOutputChannels();
    const int numSamples = buffer.getNumSamples();

    for (int i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, numSamples);

    const double dryWet = static_cast<double>(apvts.getRawParameterValue("dryWet")->load()) / 100.0;

    // Store dry signal (native double)
    juce::AudioBuffer<double> dryBuffer;
    if (dryWet < 1.0)
        dryBuffer.makeCopyOf(buffer);

    // Process phase at native sample rate (native double - no conversion!)
    phaseProcessor.process(buffer);

    // Dry/Wet mix (native double - no conversion needed!)
    if (dryWet < 1.0)
    {
        const int numChannels = buffer.getNumChannels();
        for (int ch = 0; ch < numChannels; ++ch)
        {
            double* __restrict wetData = buffer.getWritePointer(ch);
            const double* __restrict dryData = dryBuffer.getReadPointer(ch);
            const double wet = dryWet;
            const double dry = 1.0 - dryWet;

            for (int i = 0; i < numSamples; ++i)
                wetData[i] = dryData[i] * dry + wetData[i] * wet;
        }
    }

    // Output gain (native double - no conversion needed!)
    const double gain = outputGainLinear.load(std::memory_order_relaxed);
    if (gain != 1.0)
    {
        const int numChannels = buffer.getNumChannels();
        for (int ch = 0; ch < numChannels; ++ch)
        {
            double* __restrict data = buffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
                data[i] *= gain;
        }
    }
}

bool PhaseCorrectorAudioProcessor::loadCSVFile(const juce::File& file)
{
    auto result = CSVParser::parse(file);

    if (result.success && !result.points.empty())
    {
        currentCurvePoints = result.points;
        phaseProcessor.updatePhaseCurve(result.points);
        curveLoaded.store(true);
        return true;
    }

    return false;
}

void PhaseCorrectorAudioProcessor::setCurvePoints(const std::vector<std::pair<double, double>>& points)
{
    if (!points.empty())
    {
        currentCurvePoints = points;
        phaseProcessor.updatePhaseCurve(points);
        curveLoaded.store(true);
    }
}

void PhaseCorrectorAudioProcessor::clearCurve()
{
    currentCurvePoints.clear();
    phaseProcessor.updatePhaseCurve({});
    curveLoaded.store(false);
}

void PhaseCorrectorAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();

    if (!currentCurvePoints.empty())
    {
        juce::String curveData;
        for (const auto& point : currentCurvePoints)
            curveData += juce::String(point.first, 6) + ";" + juce::String(point.second, 6) + "\n";
        state.setProperty("curveData", curveData, nullptr);
    }

    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void PhaseCorrectorAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

    if (xmlState != nullptr && xmlState->hasTagName(apvts.state.getType()))
    {
        auto state = juce::ValueTree::fromXml(*xmlState);
        apvts.replaceState(state);

        juce::String curveData = state.getProperty("curveData", "");
        if (curveData.isNotEmpty())
        {
            auto result = CSVParser::parseString(curveData);
            if (result.success)
            {
                currentCurvePoints = result.points;
                phaseProcessor.updatePhaseCurve(result.points);
                curveLoaded.store(true);
            }
        }
    }
}

bool PhaseCorrectorAudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* PhaseCorrectorAudioProcessor::createEditor()
{
    return new PhaseCorrectorAudioProcessorEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new PhaseCorrectorAudioProcessor();
}
