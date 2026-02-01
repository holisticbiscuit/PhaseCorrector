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
    : fft(FFT_ORDER + 1) // FFT_SIZE = 8192
{
    // Initialize Hann windows
    for (int i = 0; i < ANALYSIS_SIZE; ++i)
    {
        float windowValue = 0.5f * (1.0f - std::cos(2.0f * juce::MathConstants<float>::pi * i / (ANALYSIS_SIZE - 1)));
        analysisWindow[i] = windowValue;
        synthesisWindow[i] = windowValue;
    }

    // Window compensation for 4x overlap with Hann
    windowCompensation = 2.0f / 3.0f * 2.0f;

    phaseTable.resize(FFT_SIZE / 2 + 1, 0.0f);
}

void PhaseProcessor::prepare(double sampleRate, int /*maxBlockSize*/)
{
    currentSampleRate = sampleRate;
    reset();
    rebuildPhaseTable();
}

void PhaseProcessor::reset()
{
    for (auto& ch : channels)
    {
        std::fill(ch.inputBuffer.begin(), ch.inputBuffer.end(), 0.0f);
        std::fill(ch.outputBuffer.begin(), ch.outputBuffer.end(), 0.0f);
        ch.inputWritePos = 0;
        ch.outputReadPos = 0;
        ch.samplesUntilNextFrame = HOP_SIZE;
    }
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
    const int numBins = FFT_SIZE / 2 + 1;
    const float depth = phaseDepth.load();

    for (int bin = 0; bin < numBins; ++bin)
    {
        float freq = static_cast<float>(bin) * static_cast<float>(currentSampleRate) / static_cast<float>(FFT_SIZE);

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
    const int fadeLength = 10;
    for (int i = 0; i < fadeLength; ++i)
    {
        float fade = static_cast<float>(i) / fadeLength;
        int lowBin = static_cast<int>((MIN_FREQ / currentSampleRate) * FFT_SIZE) + i;
        int highBin = static_cast<int>((MAX_FREQ / currentSampleRate) * FFT_SIZE) - i;

        if (lowBin >= 0 && lowBin < numBins)
            phaseTable[lowBin] *= fade;
        if (highBin >= 0 && highBin < numBins)
            phaseTable[highBin] *= fade;
    }
}

void PhaseProcessor::processFrame(int channel)
{
    auto& ch = channels[channel];

    int readPos = ch.inputWritePos - ANALYSIS_SIZE;
    if (readPos < 0)
        readPos += static_cast<int>(ch.inputBuffer.size());

    // Prepare FFT buffer
    alignas(16) std::array<float, FFT_SIZE * 2> fftData{};

    for (int i = 0; i < ANALYSIS_SIZE; ++i)
    {
        int idx = (readPos + i) % static_cast<int>(ch.inputBuffer.size());
        fftData[i] = ch.inputBuffer[idx] * analysisWindow[i];
    }

    // Forward FFT
    fft.performRealOnlyForwardTransform(fftData.data());

    // Apply phase modification
    const int numBins = FFT_SIZE / 2;
    for (int bin = 1; bin < numBins; ++bin)
    {
        float real = fftData[bin * 2];
        float imag = fftData[bin * 2 + 1];

        float magnitude = std::sqrt(real * real + imag * imag);
        float phase = std::atan2(imag, real);

        phase += phaseTable[bin];

        fftData[bin * 2] = magnitude * std::cos(phase);
        fftData[bin * 2 + 1] = magnitude * std::sin(phase);
    }

    // Inverse FFT
    fft.performRealOnlyInverseTransform(fftData.data());

    // Overlap-add
    int writePos = ch.outputReadPos;

    for (int i = 0; i < ANALYSIS_SIZE; ++i)
    {
        int idx = (writePos + i) % static_cast<int>(ch.outputBuffer.size());
        float windowedSample = fftData[i] * synthesisWindow[i] * windowCompensation;
        ch.outputBuffer[idx] += windowedSample;
    }
}

void PhaseProcessor::process(juce::AudioBuffer<float>& buffer)
{
    const int numChannels = std::min(buffer.getNumChannels(), 2);
    const int numSamples = buffer.getNumSamples();

    for (int sample = 0; sample < numSamples; ++sample)
    {
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto& state = channels[ch];

            state.inputBuffer[state.inputWritePos] = buffer.getSample(ch, sample);

            float outputSample = state.outputBuffer[state.outputReadPos];
            state.outputBuffer[state.outputReadPos] = 0.0f;

            buffer.setSample(ch, sample, outputSample);

            state.inputWritePos = (state.inputWritePos + 1) % static_cast<int>(state.inputBuffer.size());
            state.outputReadPos = (state.outputReadPos + 1) % static_cast<int>(state.outputBuffer.size());

            state.samplesUntilNextFrame--;
            if (state.samplesUntilNextFrame <= 0)
            {
                processFrame(ch);
                state.samplesUntilNextFrame = HOP_SIZE;
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
    createOversampler();
}

void OversamplingManager::createOversampler()
{
    int order = static_cast<int>(currentRate);

    if (order == 0)
    {
        oversampler.reset();
        return;
    }

    // Always use FIR filters for maximum quality (linear phase, no ringing)
    // filterHalfBandFIREquiripple provides the best stopband rejection
    oversampler = std::make_unique<juce::dsp::Oversampling<float>>(
        2, order, juce::dsp::Oversampling<float>::filterHalfBandFIREquiripple, true);
}

void OversamplingManager::prepare(double sampleRate, int blockSize)
{
    baseSampleRate = sampleRate;
    baseBlockSize = blockSize;

    createOversampler();

    if (oversampler)
        oversampler->initProcessing(static_cast<size_t>(blockSize));
}

void OversamplingManager::reset()
{
    if (oversampler)
        oversampler->reset();
}

void OversamplingManager::setRate(Rate newRate)
{
    if (newRate != currentRate)
    {
        currentRate = newRate;
        createOversampler();
        if (oversampler)
            oversampler->initProcessing(static_cast<size_t>(baseBlockSize));
    }
}

juce::dsp::AudioBlock<float> OversamplingManager::processSamplesUp(juce::dsp::AudioBlock<float>& inputBlock)
{
    if (!oversampler || currentRate == Rate::x1)
        return inputBlock;

    return oversampler->processSamplesUp(inputBlock);
}

void OversamplingManager::processSamplesDown(juce::dsp::AudioBlock<float>& outputBlock)
{
    if (!oversampler || currentRate == Rate::x1)
        return;

    oversampler->processSamplesDown(outputBlock);
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
    apvts.addParameterListener("dryWet", this);
    apvts.addParameterListener("outputGain", this);
    apvts.addParameterListener("depth", this);
    apvts.addParameterListener("nyquistFreq", this);
    apvts.addParameterListener("nyquistQ", this);
    apvts.addParameterListener("nyquistSlope", this);
    apvts.addParameterListener("nyquistEnabled", this);
}

PhaseCorrectorAudioProcessor::~PhaseCorrectorAudioProcessor()
{
    apvts.removeParameterListener("oversample", this);
    apvts.removeParameterListener("dryWet", this);
    apvts.removeParameterListener("outputGain", this);
    apvts.removeParameterListener("depth", this);
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

    // Phase Depth (like MFreeformPhase)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("depth", 1), "Depth",
        juce::NormalisableRange<float>(0.0f, 200.0f, 0.1f), 100.0f,
        juce::AudioParameterFloatAttributes().withLabel("%")));

    // Nyquist Filter Enable
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID("nyquistEnabled", 1), "Nyquist Filter", true));

    // Nyquist Frequency
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("nyquistFreq", 1), "Nyquist Freq",
        juce::NormalisableRange<float>(1000.0f, 22000.0f, 1.0f, 0.3f), 18000.0f,
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
        oversamplingManager.setRate(static_cast<OversamplingManager::Rate>(static_cast<int>(newValue)));
        double effectiveSampleRate = lastSampleRate * oversamplingManager.getFactor();
        phaseProcessor.prepare(effectiveSampleRate, lastBlockSize * oversamplingManager.getFactor());
        nyquistFilter.prepare(effectiveSampleRate);
    }
    else if (parameterID == "outputGain")
    {
        outputGain.setGainDecibels(newValue);
    }
    else if (parameterID == "depth")
    {
        phaseProcessor.setDepth(newValue / 100.0f);
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
