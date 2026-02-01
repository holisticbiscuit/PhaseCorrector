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
// Nyquist Filter Implementation
//==============================================================================
NyquistFilter::NyquistFilter()
{
    reset();
}

void NyquistFilter::prepare(double sampleRate)
{
    currentSampleRate = sampleRate;
    reset();
    updateCoefficients();
}

void NyquistFilter::reset()
{
    for (auto& channelStates : states)
        for (auto& state : channelStates)
            state = BiquadState{};
}

void NyquistFilter::setParameters(float frequency, float q, int stages)
{
    cutoffFreq = juce::jlimit(100.0f, static_cast<float>(currentSampleRate * 0.45), frequency);
    resonance = juce::jlimit(0.1f, 10.0f, q);
    numStages = juce::jlimit(1, MAX_STAGES, stages);
    updateCoefficients();
}

void NyquistFilter::updateCoefficients()
{
    // Butterworth low-pass biquad coefficients
    double w0 = 2.0 * juce::MathConstants<double>::pi * cutoffFreq / currentSampleRate;
    double cosw0 = std::cos(w0);
    double sinw0 = std::sin(w0);
    double alpha = sinw0 / (2.0 * resonance);

    double a0 = 1.0 + alpha;
    b0 = ((1.0 - cosw0) / 2.0) / a0;
    b1 = (1.0 - cosw0) / a0;
    b2 = ((1.0 - cosw0) / 2.0) / a0;
    a1 = (-2.0 * cosw0) / a0;
    a2 = (1.0 - alpha) / a0;
}

float NyquistFilter::processSample(float sample, int channel)
{
    if (channel < 0 || channel >= 2)
        return sample;

    double x = static_cast<double>(sample);

    for (int stage = 0; stage < numStages; ++stage)
    {
        auto& s = states[channel][stage];

        double y = b0 * x + b1 * s.x1 + b2 * s.x2 - a1 * s.y1 - a2 * s.y2;

        s.x2 = s.x1;
        s.x1 = x;
        s.y2 = s.y1;
        s.y1 = y;

        x = y;
    }

    return static_cast<float>(x);
}

float NyquistFilter::getMagnitudeAtFrequency(double freq) const
{
    if (freq <= 0.0 || freq >= currentSampleRate * 0.5)
        return 0.0f;

    double w = 2.0 * juce::MathConstants<double>::pi * freq / currentSampleRate;
    double cosw = std::cos(w);
    double cos2w = std::cos(2.0 * w);
    double sinw = std::sin(w);
    double sin2w = std::sin(2.0 * w);

    // H(e^jw) numerator and denominator
    double numReal = b0 + b1 * cosw + b2 * cos2w;
    double numImag = -b1 * sinw - b2 * sin2w;
    double denReal = 1.0 + a1 * cosw + a2 * cos2w;
    double denImag = -a1 * sinw - a2 * sin2w;

    double numMag = std::sqrt(numReal * numReal + numImag * numImag);
    double denMag = std::sqrt(denReal * denReal + denImag * denImag);

    double singleStageMag = (denMag > 1e-10) ? numMag / denMag : 0.0;

    // Cascade stages
    return static_cast<float>(std::pow(singleStageMag, numStages));
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
    return { "Low (2048)", "Medium (4096)", "High (8192)", "Very High (16384)", "Extreme (32768)" };
}

juce::StringArray PhaseProcessor::getOverlapNames()
{
    return { "50% (Low Latency)", "75% (High Quality)" };
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

    if (currentOverlap == Overlap::Percent75)
    {
        // Hann window for 75% overlap - COLA compliant with sum = 1.5
        for (int i = 0; i < analysisSize; ++i)
        {
            float windowValue = 0.5f * (1.0f - std::cos(2.0f * juce::MathConstants<float>::pi * i / (analysisSize - 1)));
            analysisWindow[i] = windowValue;
            synthesisWindow[i] = windowValue;
        }
        windowCompensation = 1.0f / 1.5f;  // Hann^2 sum at 75% overlap = 1.5
    }
    else
    {
        // Sine window (sqrt-Hann) for 50% overlap - COLA compliant with sum = 1.0
        // sin^2(x) + cos^2(x) = 1, so perfect reconstruction
        for (int i = 0; i < analysisSize; ++i)
        {
            float windowValue = std::sin(juce::MathConstants<float>::pi * i / (analysisSize - 1));
            analysisWindow[i] = windowValue;
            synthesisWindow[i] = windowValue;
        }
        windowCompensation = 1.0f;  // Sine^2 sum at 50% overlap = 1.0
    }
}

void PhaseProcessor::reconfigure()
{
    juce::SpinLock::ScopedLockType lock(processingLock);

    // Calculate sizes based on quality
    // Analysis order determines the analysis window size
    int analysisOrder;
    switch (currentQuality)
    {
        case Quality::Low:      analysisOrder = 10; break;  // 1024
        case Quality::Medium:   analysisOrder = 11; break;  // 2048
        case Quality::High:     analysisOrder = 12; break;  // 4096
        case Quality::VeryHigh: analysisOrder = 13; break;  // 8192
        case Quality::Extreme:  analysisOrder = 14; break;  // 16384
        default:                analysisOrder = 12; break;
    }

    analysisSize = 1 << analysisOrder;
    fftSize = analysisSize * 2;  // 2x zero-padding
    fftOrder = analysisOrder + 1;

    // Calculate hop size based on overlap
    if (currentOverlap == Overlap::Percent75)
    {
        hopSize = analysisSize / 4;  // 75% overlap = 4x
        numOverlaps = 4;
    }
    else
    {
        hopSize = analysisSize / 2;  // 50% overlap = 2x
        numOverlaps = 2;
    }

    // Recreate FFT
    fft = std::make_unique<juce::dsp::FFT>(fftOrder);

    // Rebuild windows
    buildWindows();

    // Resize buffers - need extra space for overlap-add
    int bufferSize = fftSize * 2;  // Extra space for safety
    for (auto& ch : channels)
        ch.resize(bufferSize, fftSize, hopSize);

    // Resize phase table
    phaseTable.resize(fftSize / 2 + 1, 0.0f);

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
        std::fill(phaseTable.begin(), phaseTable.end(), 0.0f);
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
        std::fill(phaseTable.begin(), phaseTable.end(), 0.0f);
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
    const float depth = phaseDepth.load();

    // Ensure phase table is correct size
    if (static_cast<int>(phaseTable.size()) != numBins)
        phaseTable.resize(numBins, 0.0f);

    for (int bin = 0; bin < numBins; ++bin)
    {
        float freq = static_cast<float>(bin) * static_cast<float>(currentSampleRate) / static_cast<float>(fftSize);

        if (freq < MIN_FREQ || freq > MAX_FREQ || !phaseCurve.isValid())
        {
            phaseTable[bin] = 0.0f;
            continue;
        }

        float logFreq = std::log10(std::max(freq, MIN_FREQ));
        double normalizedPhase = phaseCurve.evaluate(logFreq);

        // Clamp phase value
        normalizedPhase = juce::jlimit(-1.0, 1.0, normalizedPhase);

        // Convert to radians and apply depth
        float phaseRadians = static_cast<float>(normalizedPhase) * 2.0f * juce::MathConstants<float>::pi * depth;
        phaseTable[bin] = phaseRadians;
    }

    // Smooth transition at boundaries
    const int fadeLength = std::max(10, fftSize / 512);  // Scale fade with FFT size
    for (int i = 0; i < fadeLength; ++i)
    {
        float fade = static_cast<float>(i) / fadeLength;
        int lowBin = static_cast<int>((MIN_FREQ / currentSampleRate) * fftSize) + i;
        int highBin = static_cast<int>((MAX_FREQ / currentSampleRate) * fftSize) - i;

        if (lowBin >= 0 && lowBin < numBins)
            phaseTable[lowBin] *= fade;
        if (highBin >= 0 && highBin < numBins)
            phaseTable[highBin] *= fade;
    }
}

void PhaseProcessor::processFrame(int channel)
{
    auto& ch = channels[channel];

    int bufferSize = static_cast<int>(ch.inputBuffer.size());
    int readPos = ch.inputWritePos - analysisSize;
    if (readPos < 0)
        readPos += bufferSize;

    // Use pre-allocated FFT buffer
    float* fftData = ch.fftBuffer.data();

    // Clear and apply analysis window
    std::memset(fftData, 0, fftSize * 2 * sizeof(float));
    for (int i = 0; i < analysisSize; ++i)
    {
        int idx = (readPos + i) % bufferSize;
        fftData[i] = ch.inputBuffer[idx] * analysisWindow[i];
    }

    // Forward FFT
    fft->performRealOnlyForwardTransform(fftData);

    // Apply phase modification - optimized loop
    const int numBins = fftSize / 2;
    const float* phasePtr = phaseTable.data();

    for (int bin = 1; bin < numBins; ++bin)
    {
        const int idx = bin * 2;
        const float real = fftData[idx];
        const float imag = fftData[idx + 1];

        const float magnitude = std::sqrt(real * real + imag * imag);
        const float phase = std::atan2(imag, real) + phasePtr[bin];

        fftData[idx] = magnitude * std::cos(phase);
        fftData[idx + 1] = magnitude * std::sin(phase);
    }

    // Inverse FFT
    fft->performRealOnlyInverseTransform(fftData);

    // Overlap-add with synthesis window
    int writePos = ch.outputReadPos;
    const float compensation = windowCompensation;

    for (int i = 0; i < analysisSize; ++i)
    {
        int idx = (writePos + i) % bufferSize;
        ch.outputBuffer[idx] += fftData[i] * synthesisWindow[i] * compensation;
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

    for (int sample = 0; sample < numSamples; ++sample)
    {
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto& state = channels[ch];
            const int bufferSize = static_cast<int>(state.inputBuffer.size());

            state.inputBuffer[state.inputWritePos] = buffer.getSample(ch, sample);

            float outputSample = state.outputBuffer[state.outputReadPos];
            state.outputBuffer[state.outputReadPos] = 0.0f;

            buffer.setSample(ch, sample, outputSample);

            state.inputWritePos = (state.inputWritePos + 1) % bufferSize;
            state.outputReadPos = (state.outputReadPos + 1) % bufferSize;

            state.samplesUntilNextFrame--;
            if (state.samplesUntilNextFrame <= 0)
            {
                processFrame(ch);
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
// Oversampling Manager Implementation
//==============================================================================
OversamplingManager::OversamplingManager()
{
    // Don't create oversampler in constructor - wait for prepare()
}

void OversamplingManager::createOversampler()
{
    // This should only be called from audio thread or during prepare
    int order = static_cast<int>(currentRate);

    // Reset and release old oversampler first
    if (oversampler)
    {
        oversampler->reset();
        oversampler.reset();
    }

    if (order == 0)
        return;

    // Select filter type based on mode
    auto filterType = (filterMode == FilterMode::FIR)
        ? juce::dsp::Oversampling<float>::filterHalfBandFIREquiripple
        : juce::dsp::Oversampling<float>::filterHalfBandPolyphaseIIR;

    try
    {
        // Create new oversampler with 2 channels
        oversampler = std::make_unique<juce::dsp::Oversampling<float>>(2, order, filterType, true);

        // Initialize with current block size
        if (baseBlockSize > 0)
        {
            size_t maxBlockSize = static_cast<size_t>(baseBlockSize);
            oversampler->initProcessing(maxBlockSize);
        }
    }
    catch (const std::exception&)
    {
        DBG("Oversampler creation failed");
        oversampler.reset();
        currentRate = Rate::x1;
    }
    catch (...)
    {
        DBG("Oversampler creation failed with unknown error");
        oversampler.reset();
        currentRate = Rate::x1;
    }
}

void OversamplingManager::prepare(double sampleRate, int blockSize)
{
    baseSampleRate = sampleRate;
    baseBlockSize = blockSize;
    isPrepared = false;

    // Clear any pending changes
    pendingRate.store(-1);
    pendingFilterMode.store(-1);

    createOversampler();

    isPrepared = true;
}

void OversamplingManager::reset()
{
    if (oversampler)
        oversampler->reset();
}

void OversamplingManager::setRate(Rate newRate)
{
    // Just store the pending change - it will be applied on audio thread
    pendingRate.store(static_cast<int>(newRate));
}

void OversamplingManager::setFilterMode(FilterMode mode)
{
    // Just store the pending change - it will be applied on audio thread
    pendingFilterMode.store(static_cast<int>(mode));
}

bool OversamplingManager::applyPendingChanges()
{
    // This is called at the start of processBlock, on the audio thread
    bool needsRecreate = false;

    int newRate = pendingRate.exchange(-1);
    if (newRate >= 0 && static_cast<Rate>(newRate) != currentRate)
    {
        currentRate = static_cast<Rate>(newRate);
        needsRecreate = true;
    }

    int newMode = pendingFilterMode.exchange(-1);
    if (newMode >= 0 && static_cast<FilterMode>(newMode) != filterMode)
    {
        filterMode = static_cast<FilterMode>(newMode);
        if (currentRate != Rate::x1)
            needsRecreate = true;
    }

    if (needsRecreate && isPrepared)
    {
        createOversampler();
    }

    return needsRecreate;
}

juce::dsp::AudioBlock<float> OversamplingManager::processSamplesUp(juce::dsp::AudioBlock<float>& inputBlock)
{
    if (!isPrepared || !oversampler || currentRate == Rate::x1)
        return inputBlock;

    try
    {
        return oversampler->processSamplesUp(inputBlock);
    }
    catch (...)
    {
        return inputBlock;
    }
}

void OversamplingManager::processSamplesDown(juce::dsp::AudioBlock<float>& outputBlock)
{
    if (!isPrepared || !oversampler || currentRate == Rate::x1)
        return;

    try
    {
        oversampler->processSamplesDown(outputBlock);
    }
    catch (...)
    {
        // Silently fail - audio passes through unprocessed
    }
}

float OversamplingManager::getLatencySamples() const
{
    if (!oversampler || currentRate == Rate::x1)
        return 0.0f;

    return oversampler->getLatencyInSamples();
}

juce::StringArray OversamplingManager::getRateNames()
{
    return { "1x (Off)", "2x", "4x", "8x", "16x", "32x", "64x" };
}

juce::StringArray OversamplingManager::getFilterModeNames()
{
    return { "Linear Phase (FIR)", "Minimum Phase (IIR)" };
}

//==============================================================================
// Plugin Processor Implementation
//==============================================================================
PhaseCorrectorAudioProcessor::PhaseCorrectorAudioProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    apvts.addParameterListener("oversample", this);
    apvts.addParameterListener("ovsMode", this);
    apvts.addParameterListener("dryWet", this);
    apvts.addParameterListener("outputGain", this);
    apvts.addParameterListener("depth", this);
    apvts.addParameterListener("fftQuality", this);
    apvts.addParameterListener("fftOverlap", this);
    apvts.addParameterListener("nyquistFreq", this);
    apvts.addParameterListener("nyquistQ", this);
    apvts.addParameterListener("nyquistSlope", this);
    apvts.addParameterListener("nyquistEnabled", this);
}

PhaseCorrectorAudioProcessor::~PhaseCorrectorAudioProcessor()
{
    apvts.removeParameterListener("oversample", this);
    apvts.removeParameterListener("ovsMode", this);
    apvts.removeParameterListener("dryWet", this);
    apvts.removeParameterListener("outputGain", this);
    apvts.removeParameterListener("depth", this);
    apvts.removeParameterListener("fftQuality", this);
    apvts.removeParameterListener("fftOverlap", this);
    apvts.removeParameterListener("nyquistFreq", this);
    apvts.removeParameterListener("nyquistQ", this);
    apvts.removeParameterListener("nyquistSlope", this);
    apvts.removeParameterListener("nyquistEnabled", this);
}

juce::AudioProcessorValueTreeState::ParameterLayout PhaseCorrectorAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Oversampling (0-6 for 1x to 64x)
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID("oversample", 1), "Oversampling",
        OversamplingManager::getRateNames(), 2)); // Default 4x

    // Oversampling filter mode (FIR = linear phase, IIR = minimum phase)
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID("ovsMode", 1), "OVS Mode",
        OversamplingManager::getFilterModeNames(), 0)); // Default FIR

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
        PhaseProcessor::getOverlapNames(), 1));  // Default: 75% (High Quality)

    // Nyquist Filter Enable
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID("nyquistEnabled", 1), "Nyquist Filter", true));

    // Nyquist Frequency (extended to 40kHz for high sample rates)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("nyquistFreq", 1), "Nyquist Freq",
        juce::NormalisableRange<float>(1000.0f, 40000.0f, 1.0f, 0.3f), 18000.0f,
        juce::AudioParameterFloatAttributes().withLabel("Hz")));

    // Nyquist Q
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("nyquistQ", 1), "Nyquist Q",
        juce::NormalisableRange<float>(0.5f, 5.0f, 0.01f), 0.707f));

    // Nyquist Slope (dB/octave)
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID("nyquistSlope", 1), "Nyquist Slope",
        juce::StringArray{ "12 dB/oct", "24 dB/oct", "36 dB/oct", "48 dB/oct", "60 dB/oct", "72 dB/oct", "84 dB/oct", "96 dB/oct" },
        1)); // Default 24 dB/oct

    return { params.begin(), params.end() };
}

void PhaseCorrectorAudioProcessor::parameterChanged(const juce::String& parameterID, float newValue)
{
    if (parameterID == "oversample")
    {
        // Just set the pending rate - actual change happens on audio thread
        oversamplingManager.setRate(static_cast<OversamplingManager::Rate>(static_cast<int>(newValue)));
    }
    else if (parameterID == "ovsMode")
    {
        // Just set the pending mode - actual change happens on audio thread
        oversamplingManager.setFilterMode(static_cast<OversamplingManager::FilterMode>(static_cast<int>(newValue)));
    }
    else if (parameterID == "outputGain")
    {
        outputGain.setGainDecibels(newValue);
    }
    else if (parameterID == "depth")
    {
        phaseProcessor.setDepth(newValue / 100.0f);
    }
    else if (parameterID == "fftQuality")
    {
        phaseProcessor.setQuality(static_cast<PhaseProcessor::Quality>(static_cast<int>(newValue)));
        // Update latency reporting to host
        setLatencySamples(phaseProcessor.getLatencySamples() +
                          static_cast<int>(oversamplingManager.getLatencySamples()));
    }
    else if (parameterID == "fftOverlap")
    {
        phaseProcessor.setOverlap(static_cast<PhaseProcessor::Overlap>(static_cast<int>(newValue)));
        // Update latency reporting to host
        setLatencySamples(phaseProcessor.getLatencySamples() +
                          static_cast<int>(oversamplingManager.getLatencySamples()));
    }
    else if (parameterID == "nyquistFreq" || parameterID == "nyquistQ" || parameterID == "nyquistSlope")
    {
        float freq = apvts.getRawParameterValue("nyquistFreq")->load();
        float q = apvts.getRawParameterValue("nyquistQ")->load();
        int slope = static_cast<int>(apvts.getRawParameterValue("nyquistSlope")->load()) + 1;
        nyquistFilter.setParameters(freq, q, slope);
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

    oversamplingManager.prepare(sampleRate, samplesPerBlock);

    double effectiveSampleRate = sampleRate * oversamplingManager.getFactor();
    phaseProcessor.prepare(effectiveSampleRate, samplesPerBlock * oversamplingManager.getFactor());
    nyquistFilter.prepare(effectiveSampleRate);

    // Apply current parameter values
    float freq = apvts.getRawParameterValue("nyquistFreq")->load();
    float q = apvts.getRawParameterValue("nyquistQ")->load();
    int slope = static_cast<int>(apvts.getRawParameterValue("nyquistSlope")->load()) + 1;
    nyquistFilter.setParameters(freq, q, slope);

    phaseProcessor.setDepth(apvts.getRawParameterValue("depth")->load() / 100.0f);

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels = 2;
    outputGain.prepare(spec);
    outputGain.setGainDecibels(apvts.getRawParameterValue("outputGain")->load());

    // Report total latency to host for PDC
    setLatencySamples(phaseProcessor.getLatencySamples() +
                      static_cast<int>(oversamplingManager.getLatencySamples()));
}

void PhaseCorrectorAudioProcessor::releaseResources()
{
    oversamplingManager.reset();
    phaseProcessor.reset();
    nyquistFilter.reset();
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

    // Apply any pending oversampling changes safely on audio thread
    if (oversamplingManager.applyPendingChanges())
    {
        // Oversampling rate changed - update dependent processors
        double effectiveSampleRate = lastSampleRate * oversamplingManager.getFactor();
        int effectiveBlockSize = lastBlockSize * oversamplingManager.getFactor();
        phaseProcessor.prepare(effectiveSampleRate, effectiveBlockSize);
        nyquistFilter.prepare(effectiveSampleRate);

        // Update latency reporting
        setLatencySamples(phaseProcessor.getLatencySamples() +
                          static_cast<int>(oversamplingManager.getLatencySamples()));
    }

    const int totalNumInputChannels = getTotalNumInputChannels();
    const int totalNumOutputChannels = getTotalNumOutputChannels();

    for (int i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    const float dryWet = apvts.getRawParameterValue("dryWet")->load() / 100.0f;
    const bool nyquistEnabled = apvts.getRawParameterValue("nyquistEnabled")->load() > 0.5f;

    // Store dry signal
    juce::AudioBuffer<float> dryBuffer;
    if (dryWet < 1.0f)
        dryBuffer.makeCopyOf(buffer);

    // Upsample
    juce::dsp::AudioBlock<float> block(buffer);
    auto oversampledBlock = oversamplingManager.processSamplesUp(block);

    // Create buffer wrapper for oversampled data
    const int numChannels = static_cast<int>(oversampledBlock.getNumChannels());
    const int numSamples = static_cast<int>(oversampledBlock.getNumSamples());

    float* channelPtrs[2] = { nullptr, nullptr };
    for (int ch = 0; ch < numChannels && ch < 2; ++ch)
        channelPtrs[ch] = oversampledBlock.getChannelPointer(ch);

    juce::AudioBuffer<float> oversampledBuffer(channelPtrs, numChannels, numSamples);

    // Apply Nyquist filter before phase processing
    if (nyquistEnabled)
    {
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto* data = oversampledBuffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
                data[i] = nyquistFilter.processSample(data[i], ch);
        }
    }

    // Process phase
    phaseProcessor.process(oversampledBuffer);

    // Downsample
    oversamplingManager.processSamplesDown(block);

    // Dry/Wet mix
    if (dryWet < 1.0f)
    {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            auto* wetData = buffer.getWritePointer(ch);
            const auto* dryData = dryBuffer.getReadPointer(ch);

            for (int i = 0; i < buffer.getNumSamples(); ++i)
                wetData[i] = dryData[i] * (1.0f - dryWet) + wetData[i] * dryWet;
        }
    }

    // Output gain
    juce::dsp::AudioBlock<float> outputBlock(buffer);
    juce::dsp::ProcessContextReplacing<float> context(outputBlock);
    outputGain.process(context);
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
