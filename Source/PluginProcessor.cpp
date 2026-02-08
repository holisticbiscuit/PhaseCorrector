/*
  ==============================================================================
    PhaseCorrector - Plugin Processor Implementation
    Overlap-Add FFT processing with cubic spline phase interpolation
    Optimized: SoA FFT, AVX2 SIMD, block I/O, bitmask addressing
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
// Double-Precision FFT Implementation using FFTW3
// Split-radix, cache-oblivious, AVX2-native
//==============================================================================
DoubleFFT::DoubleFFT(int order)
{
    initialize(order);
}

DoubleFFT::~DoubleFFT()
{
    cleanup();
}

void DoubleFFT::cleanup()
{
    if (forwardPlan) { fftw_destroy_plan(forwardPlan); forwardPlan = nullptr; }
    if (inversePlan) { fftw_destroy_plan(inversePlan); inversePlan = nullptr; }
    if (fftwReal) { fftw_free(fftwReal); fftwReal = nullptr; }
    if (fftwComplex) { fftw_free(fftwComplex); fftwComplex = nullptr; }
}

void DoubleFFT::initialize(int order)
{
    cleanup();

    fftOrder = order;
    fftSize = 1 << order;

    // Allocate FFTW-aligned buffers
    fftwReal = fftw_alloc_real(fftSize);
    fftwComplex = fftw_alloc_complex(fftSize / 2 + 1);

    // Use FFTW_ESTIMATE for instant plan creation (no benchmarking)
    // FFTW_MEASURE blocks for seconds on large FFTs and crashes hosts with audio timeouts
    // FFTW_ESTIMATE picks a good plan heuristically — same output, ~10-20% slower execution
    forwardPlan = fftw_plan_dft_r2c_1d(fftSize, fftwReal, fftwComplex, FFTW_ESTIMATE);
    inversePlan = fftw_plan_dft_c2r_1d(fftSize, fftwComplex, fftwReal, FFTW_ESTIMATE);
}

void DoubleFFT::performRealForward(const double* input, double* outReal, double* outImag)
{
    // Copy input into FFTW-aligned buffer (FFTW may modify input with MEASURE plans)
    std::memcpy(fftwReal, input, fftSize * sizeof(double));

    fftw_execute(forwardPlan);

    // Deinterleave FFTW's complex output [re,im,re,im,...] into SoA format
    const int numBins = fftSize / 2 + 1;
    for (int i = 0; i < numBins; ++i)
    {
        outReal[i] = fftwComplex[i][0];
        outImag[i] = fftwComplex[i][1];
    }
}

void DoubleFFT::performRealInverse(const double* inReal, const double* inImag, double* output)
{
    // Interleave SoA format into FFTW's complex input [re,im,re,im,...]
    const int numBins = fftSize / 2 + 1;
    for (int i = 0; i < numBins; ++i)
    {
        fftwComplex[i][0] = inReal[i];
        fftwComplex[i][1] = inImag[i];
    }

    // FFTW c2r destroys fftwComplex (expected — we've already consumed it)
    fftw_execute(inversePlan);

    // Copy result and normalize (FFTW does NOT normalize inverse transforms)
    const double scale = 1.0 / fftSize;
    for (int i = 0; i < fftSize; ++i)
        output[i] = fftwReal[i] * scale;
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
    return { "50% (2x)", "75% (4x)", "87.5% (8x)", "93.75% (16x)", "96.875% (32x)", "98.4% (64x)", "99.2% (128x)", "99.6% (256x)" };
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
    // Use periodic Hann window (N in denominator) for proper COLA
    for (int i = 0; i < analysisSize; ++i)
    {
        double windowValue = 0.5 * (1.0 - std::cos(2.0 * juce::MathConstants<double>::pi * i / analysisSize));
        analysisWindow[i] = windowValue;
        synthesisWindow[i] = windowValue;
    }

    // Compute COLA compensation from window energy using Kahan summation
    double windowSquaredSum = 0.0;
    double windowSumError = 0.0;
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
            hopSize = analysisSize / 2;
            numOverlaps = 2;
            break;
        case Overlap::Percent75:
            hopSize = analysisSize / 4;
            numOverlaps = 4;
            break;
        case Overlap::Percent875:
            hopSize = analysisSize / 8;
            numOverlaps = 8;
            break;
        case Overlap::Percent9375:
            hopSize = analysisSize / 16;
            numOverlaps = 16;
            break;
        case Overlap::Percent96875:
            hopSize = analysisSize / 32;
            numOverlaps = 32;
            break;
        case Overlap::Percent984375:
            hopSize = analysisSize / 64;
            numOverlaps = 64;
            break;
        case Overlap::Percent9921875:
            hopSize = analysisSize / 128;
            numOverlaps = 128;
            break;
        case Overlap::Percent99609375:
            hopSize = analysisSize / 256;
            numOverlaps = 256;
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
    // Buffer size must be power of 2 for bitmask addressing
    // Use 4x for safety margin with high overlap modes and large host blocks
    int bufferSize = fftSize * 4;
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
        filterIRReady.store(false);
        needsFilterRebuild.store(true);
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
        filterIRReady.store(false);
        needsFilterRebuild.store(true);
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
        phaseCurve.clear();
    }

    // Defer the actual filter rebuild to the audio thread (avoids data race on filterSpec arrays)
    needsFilterRebuild.store(true);
}

void PhaseProcessor::rebuildPhaseTable()
{
    const int numBins = fftSize / 2 + 1;
    const double depth = phaseDepth.load();

    // Ensure phase table is correct size
    if (static_cast<int>(phaseTable.size()) != numBins)
        phaseTable.resize(numBins, 0.0);

    // Also rebuild the filter spectrum for proper all-pass filtering
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
    // Store as separate real/imag arrays (SoA) for SIMD complex multiply

    const int numBins = fftSize / 2 + 1;
    const double depth = phaseDepth.load();

    // Resize filter spectrum buffers
    if (static_cast<int>(filterSpecReal.size()) != numBins)
    {
        filterSpecReal.resize(numBins, 0.0);
        filterSpecImag.resize(numBins, 0.0);
    }

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
        filterSpecReal[bin] = std::cos(phase);
        filterSpecImag[bin] = std::sin(phase);
    }

    filterIRReady.store(true);
}

void PhaseProcessor::processFrame(int channel)
{
    auto& ch = channels[channel];

    const int bufSize = static_cast<int>(ch.inputBuffer.size());
    const int mask = ch.bufferMask;
    int readPos = ch.inputWritePos - analysisSize;
    if (readPos < 0)
        readPos += bufSize;

    // Fill FFT real buffer with windowed input samples
    double* __restrict fftR = ch.fftReal.data();
    double* __restrict fftI = ch.fftImag.data();
    const double* __restrict inBuf = ch.inputBuffer.data();
    const double* __restrict anaWin = analysisWindow.data();

    if (readPos + analysisSize <= bufSize)
    {
        // Fast path: no wrap-around
        for (int i = 0; i < analysisSize; ++i)
            fftR[i] = inBuf[readPos + i] * anaWin[i];
    }
    else
    {
        // Wrap-around path with bitmask
        for (int i = 0; i < analysisSize; ++i)
            fftR[i] = inBuf[(readPos + i) & mask] * anaWin[i];
    }

    // Forward FFT (SoA: separate real/imag output)
    fft.performRealForward(fftR, fftR, fftI);

    // Apply all-pass filter via complex multiplication in frequency domain
    const double depth = std::abs(phaseDepth.load());
    const bool hasPhaseData = phaseCurve.isValid() && depth > 0.001 && filterIRReady.load();

    if (hasPhaseData)
    {
        const double* __restrict hR = filterSpecReal.data();
        const double* __restrict hI = filterSpecImag.data();
        const int numBins = fftSize / 2 + 1;

        int bin = 0;

#ifdef PHASE_USE_AVX2
        // AVX2 path: process 4 bins at once (4 doubles per 256-bit register)
        // Produces bit-identical results to scalar (IEEE 754 compliant)
        for (; bin + 3 < numBins; bin += 4)
        {
            __m256d xr = _mm256_loadu_pd(&fftR[bin]);
            __m256d xi = _mm256_loadu_pd(&fftI[bin]);
            __m256d hr = _mm256_loadu_pd(&hR[bin]);
            __m256d hi = _mm256_loadu_pd(&hI[bin]);

            // Complex multiply: Y = X * H
            // real = xr*hr - xi*hi, imag = xr*hi + xi*hr
            __m256d yr = _mm256_sub_pd(_mm256_mul_pd(xr, hr), _mm256_mul_pd(xi, hi));
            __m256d yi = _mm256_add_pd(_mm256_mul_pd(xr, hi), _mm256_mul_pd(xi, hr));

            _mm256_storeu_pd(&fftR[bin], yr);
            _mm256_storeu_pd(&fftI[bin], yi);
        }
#endif

        // Scalar tail (and fallback when AVX2 not available)
        for (; bin < numBins; ++bin)
        {
            double xr = fftR[bin];
            double xi = fftI[bin];
            fftR[bin] = xr * hR[bin] - xi * hI[bin];
            fftI[bin] = xr * hI[bin] + xi * hR[bin];
        }
    }

    // Inverse FFT (SoA: separate real/imag input, real output into fftR)
    fft.performRealInverse(fftR, fftI, fftR);

    // Overlap-add with synthesis window using Kahan compensated summation
    const int writePos = ch.outputReadPos;
    double* __restrict outBuf = ch.outputBuffer.data();
    double* __restrict outBufErr = ch.outputBufferError.data();
    const double* __restrict synWin = synthesisWindow.data();
    const double compensation = windowCompensation;

    if (writePos + analysisSize <= bufSize)
    {
        // Fast path: no wrap-around
        for (int i = 0; i < analysisSize; ++i)
        {
            const int idx = writePos + i;
            const double value = fftR[i] * synWin[i] * compensation;
            // Kahan compensated summation for near-infinite precision accumulation
            const double y = value - outBufErr[idx];
            const double t = outBuf[idx] + y;
            outBufErr[idx] = (t - outBuf[idx]) - y;
            outBuf[idx] = t;
        }
    }
    else
    {
        // Wrap-around path with bitmask
        for (int i = 0; i < analysisSize; ++i)
        {
            const int idx = (writePos + i) & mask;
            const double value = fftR[i] * synWin[i] * compensation;
            const double y = value - outBufErr[idx];
            const double t = outBuf[idx] + y;
            outBufErr[idx] = (t - outBuf[idx]) - y;
            outBuf[idx] = t;
        }
    }
}

void PhaseProcessor::processFrameBypass(int channel)
{
    // Optimized bypass: simple delayed copy of hopSize samples
    // Each output position receives the exact same input value from all overlapping
    // frames (pure delay), so we only need to write once per position.
    auto& ch = channels[channel];

    const int bufSize = static_cast<int>(ch.inputBuffer.size());
    const int mask = ch.bufferMask;

    int readPos = ch.inputWritePos - analysisSize;
    if (readPos < 0)
        readPos += bufSize;

    const int writePos = ch.outputReadPos;

    if (readPos + hopSize <= bufSize && writePos + hopSize <= bufSize)
    {
        // Fast path: no wrap-around on either buffer
        std::memcpy(&ch.outputBuffer[writePos], &ch.inputBuffer[readPos], hopSize * sizeof(double));
    }
    else
    {
        for (int i = 0; i < hopSize; ++i)
            ch.outputBuffer[(writePos + i) & mask] = ch.inputBuffer[(readPos + i) & mask];
    }
}

//==============================================================================
// Block-Based Processing (float) - channel-first with bulk copy
//==============================================================================
void PhaseProcessor::process(juce::AudioBuffer<float>& buffer)
{
    if (needsReconfigure.load())
        reconfigure();

    juce::SpinLock::ScopedTryLockType lock(processingLock);
    if (!lock.isLocked())
        return;

    // Rebuild filter spectrum safely under processing lock (deferred from message thread)
    if (needsFilterRebuild.load())
    {
        rebuildPhaseTable();
        needsFilterRebuild.store(false);
    }

    const int numChannels = std::min(buffer.getNumChannels(), 2);
    const int numSamples = buffer.getNumSamples();

    const double depth = std::abs(phaseDepth.load());
    const bool shouldProcess = phaseCurve.isValid() && depth > 0.001;

    // Process each channel in full before moving to the next (better cache behavior)
    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto& state = channels[ch];
        const int bufSize = static_cast<int>(state.inputBuffer.size());
        const int mask = state.bufferMask;
        float* __restrict hostData = buffer.getWritePointer(ch);
        double* __restrict inBuf = state.inputBuffer.data();
        double* __restrict outBuf = state.outputBuffer.data();
        double* __restrict outErr = state.outputBufferError.data();

        int remaining = numSamples;
        int hostPos = 0;

        while (remaining > 0)
        {
            int chunk = std::min(remaining, state.samplesUntilNextFrame);

            // Copy input chunk: float host -> double circular buffer
            {
                int writeStart = state.inputWritePos;
                int firstPart = std::min(chunk, bufSize - writeStart);
                for (int i = 0; i < firstPart; ++i)
                    inBuf[writeStart + i] = static_cast<double>(hostData[hostPos + i]);
                if (firstPart < chunk)
                {
                    int secondPart = chunk - firstPart;
                    for (int i = 0; i < secondPart; ++i)
                        inBuf[i] = static_cast<double>(hostData[hostPos + firstPart + i]);
                }
            }

            // Copy output chunk: double circular buffer -> float host, then zero
            {
                int readStart = state.outputReadPos;
                int firstPart = std::min(chunk, bufSize - readStart);
                for (int i = 0; i < firstPart; ++i)
                    hostData[hostPos + i] = static_cast<float>(outBuf[readStart + i]);
                std::memset(&outBuf[readStart], 0, firstPart * sizeof(double));
                std::memset(&outErr[readStart], 0, firstPart * sizeof(double));
                if (firstPart < chunk)
                {
                    int secondPart = chunk - firstPart;
                    for (int i = 0; i < secondPart; ++i)
                        hostData[hostPos + firstPart + i] = static_cast<float>(outBuf[i]);
                    std::memset(&outBuf[0], 0, secondPart * sizeof(double));
                    std::memset(&outErr[0], 0, secondPart * sizeof(double));
                }
            }

            state.inputWritePos = (state.inputWritePos + chunk) & mask;
            state.outputReadPos = (state.outputReadPos + chunk) & mask;
            state.samplesUntilNextFrame -= chunk;
            hostPos += chunk;
            remaining -= chunk;

            if (state.samplesUntilNextFrame <= 0)
            {
                if (shouldProcess)
                    processFrame(ch);
                else
                    processFrameBypass(ch);
                state.samplesUntilNextFrame = hopSize;
            }
        }
    }
}

//==============================================================================
// Block-Based Processing (double) - native 64-bit, no float conversion
//==============================================================================
void PhaseProcessor::process(juce::AudioBuffer<double>& buffer)
{
    if (needsReconfigure.load())
        reconfigure();

    juce::SpinLock::ScopedTryLockType lock(processingLock);
    if (!lock.isLocked())
        return;

    // Rebuild filter spectrum safely under processing lock (deferred from message thread)
    if (needsFilterRebuild.load())
    {
        rebuildPhaseTable();
        needsFilterRebuild.store(false);
    }

    const int numChannels = std::min(buffer.getNumChannels(), 2);
    const int numSamples = buffer.getNumSamples();

    const double depth = std::abs(phaseDepth.load());
    const bool shouldProcess = phaseCurve.isValid() && depth > 0.001;

    // Process each channel in full before moving to the next (better cache behavior)
    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto& state = channels[ch];
        const int bufSize = static_cast<int>(state.inputBuffer.size());
        const int mask = state.bufferMask;
        double* __restrict hostData = buffer.getWritePointer(ch);
        double* __restrict inBuf = state.inputBuffer.data();
        double* __restrict outBuf = state.outputBuffer.data();
        double* __restrict outErr = state.outputBufferError.data();

        int remaining = numSamples;
        int hostPos = 0;

        while (remaining > 0)
        {
            int chunk = std::min(remaining, state.samplesUntilNextFrame);

            // Copy input chunk: host -> circular buffer (native double, use memcpy)
            {
                int writeStart = state.inputWritePos;
                int firstPart = std::min(chunk, bufSize - writeStart);
                std::memcpy(&inBuf[writeStart], &hostData[hostPos], firstPart * sizeof(double));
                if (firstPart < chunk)
                {
                    int secondPart = chunk - firstPart;
                    std::memcpy(&inBuf[0], &hostData[hostPos + firstPart], secondPart * sizeof(double));
                }
            }

            // Copy output chunk: circular buffer -> host, then zero
            {
                int readStart = state.outputReadPos;
                int firstPart = std::min(chunk, bufSize - readStart);
                std::memcpy(&hostData[hostPos], &outBuf[readStart], firstPart * sizeof(double));
                std::memset(&outBuf[readStart], 0, firstPart * sizeof(double));
                std::memset(&outErr[readStart], 0, firstPart * sizeof(double));
                if (firstPart < chunk)
                {
                    int secondPart = chunk - firstPart;
                    std::memcpy(&hostData[hostPos + firstPart], &outBuf[0], secondPart * sizeof(double));
                    std::memset(&outBuf[0], 0, secondPart * sizeof(double));
                    std::memset(&outErr[0], 0, secondPart * sizeof(double));
                }
            }

            state.inputWritePos = (state.inputWritePos + chunk) & mask;
            state.outputReadPos = (state.outputReadPos + chunk) & mask;
            state.samplesUntilNextFrame -= chunk;
            hostPos += chunk;
            remaining -= chunk;

            if (state.samplesUntilNextFrame <= 0)
            {
                if (shouldProcess)
                    processFrame(ch);
                else
                    processFrameBypass(ch);
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
        PhaseProcessor::getOverlapNames(), 5));  // Default: 98.4% (64x)

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
        phaseProcessor.setQuality(static_cast<PhaseProcessor::Quality>(static_cast<int>(newValue)));
        // Latency is updated after reconfigure() actually completes in processBlock
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

    // Sync latency with host after any reconfigure
    const int actualLatency = phaseProcessor.getLatencySamples();
    if (actualLatency != getLatencySamples())
        setLatencySamples(actualLatency);

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

    // Sync latency with host after any reconfigure
    const int actualLatency = phaseProcessor.getLatencySamples();
    if (actualLatency != getLatencySamples())
        setLatencySamples(actualLatency);

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
