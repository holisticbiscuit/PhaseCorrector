/*
  ==============================================================================
    PhaseCorrector - Freeform Phase EQ Plugin
    Low-THD FFT-based phase manipulation with 4x Overlap-Add architecture
    Inspired by MFreeformPhase with superior audio quality
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <array>
#include <vector>
#include <complex>
#include <atomic>
#include <mutex>

//==============================================================================
// Cubic Spline Interpolation for smooth phase curves
//==============================================================================
class CubicSpline
{
public:
    CubicSpline() = default;

    bool build(const std::vector<double>& x, const std::vector<double>& y);
    double evaluate(double x) const;
    bool isValid() const { return valid && !xData.empty(); }
    void clear();

private:
    std::vector<double> xData, yData;
    std::vector<double> a, b, c, d;
    bool valid = false;
};

//==============================================================================
// Nyquist Filter - Cascaded Biquad Low-Pass
//==============================================================================
class NyquistFilter
{
public:
    static constexpr int MAX_STAGES = 8; // Up to 96 dB/octave

    NyquistFilter();

    void prepare(double sampleRate);
    void reset();
    void setParameters(float frequency, float q, int stages);
    float processSample(float sample, int channel);

    // For visualization
    float getMagnitudeAtFrequency(double freq) const;
    float getFrequency() const { return cutoffFreq; }
    float getQ() const { return resonance; }
    int getStages() const { return numStages; }

private:
    void updateCoefficients();

    struct BiquadState
    {
        double x1 = 0, x2 = 0;
        double y1 = 0, y2 = 0;
    };

    std::array<std::array<BiquadState, MAX_STAGES>, 2> states; // Per channel, per stage

    // Coefficients (same for all stages)
    double b0 = 1, b1 = 0, b2 = 0;
    double a1 = 0, a2 = 0;

    double currentSampleRate = 44100.0;
    float cutoffFreq = 20000.0f;
    float resonance = 0.707f;
    int numStages = 2;
};

//==============================================================================
// FFT-based Phase Processor with Overlap-Add
//==============================================================================
class PhaseProcessor
{
public:
    static constexpr int FFT_ORDER = 12;
    static constexpr int ANALYSIS_SIZE = 1 << FFT_ORDER;    // 4096
    static constexpr int FFT_SIZE = ANALYSIS_SIZE * 2;      // 8192 (2x zero-padding)
    static constexpr int HOP_SIZE = ANALYSIS_SIZE / 4;      // 1024 (4x overlap)
    static constexpr int NUM_OVERLAPS = 4;

    PhaseProcessor();
    ~PhaseProcessor() = default;

    void prepare(double sampleRate, int maxBlockSize);
    void process(juce::AudioBuffer<float>& buffer);
    void reset();

    void updatePhaseCurve(const std::vector<std::pair<double, double>>& points);
    void setDepth(float depth) { phaseDepth.store(juce::jlimit(-2.0f, 2.0f, depth)); }
    float getDepth() const { return phaseDepth.load(); }

    const std::vector<float>& getPhaseTable() const { return phaseTable; }
    double getSampleRate() const { return currentSampleRate; }

private:
    void rebuildPhaseTable();
    void processFrame(int channel);

    juce::dsp::FFT fft;

    struct ChannelState
    {
        std::array<float, FFT_SIZE> inputBuffer{};
        std::array<float, FFT_SIZE> outputBuffer{};
        int inputWritePos = 0;
        int outputReadPos = 0;
        int samplesUntilNextFrame = HOP_SIZE;
    };

    std::array<ChannelState, 2> channels;

    // Windows (Hann for COLA compliance)
    std::array<float, ANALYSIS_SIZE> analysisWindow{};
    std::array<float, ANALYSIS_SIZE> synthesisWindow{};
    float windowCompensation = 1.0f;

    // Phase modification table
    std::vector<float> phaseTable;
    CubicSpline phaseCurve;
    std::mutex curveMutex;

    double currentSampleRate = 44100.0;
    std::atomic<float> phaseDepth{1.0f};

    static constexpr float MIN_FREQ = 20.0f;
    static constexpr float MAX_FREQ = 20000.0f;
    static constexpr float LOG_MIN_FREQ = 1.301030f;
    static constexpr float LOG_MAX_FREQ = 4.301030f;
};

//==============================================================================
// CSV Parser for phase curves
//==============================================================================
class CSVParser
{
public:
    struct ParseResult
    {
        bool success = false;
        std::vector<std::pair<double, double>> points;
        juce::String errorMessage;
        int pointCount = 0;
    };

    static ParseResult parse(const juce::File& file);
    static ParseResult parseString(const juce::String& content);

private:
    static constexpr float LOG_MIN_FREQ = 1.301030f;
    static constexpr float LOG_MAX_FREQ = 4.301030f;
};

//==============================================================================
// Oversampling Manager
//==============================================================================
class OversamplingManager
{
public:
    enum class Rate
    {
        x1 = 0,
        x2 = 1,
        x4 = 2,
        x8 = 3,
        x16 = 4,
        x32 = 5,
        x64 = 6
    };

    OversamplingManager();
    ~OversamplingManager() = default;

    void prepare(double sampleRate, int blockSize);
    void reset();

    juce::dsp::AudioBlock<float> processSamplesUp(juce::dsp::AudioBlock<float>& inputBlock);
    void processSamplesDown(juce::dsp::AudioBlock<float>& outputBlock);

    void setRate(Rate newRate);
    Rate getRate() const { return currentRate; }
    int getFactor() const { return 1 << static_cast<int>(currentRate); }
    float getLatencySamples() const;

    static juce::StringArray getRateNames();

private:
    void createOversampler();

    std::unique_ptr<juce::dsp::Oversampling<float>> oversampler;
    Rate currentRate = Rate::x1;
    Rate pendingRate = Rate::x1;
    std::atomic<bool> needsUpdate{false};

    double baseSampleRate = 44100.0;
    int baseBlockSize = 512;
};

//==============================================================================
// Main Plugin Processor
//==============================================================================
class PhaseCorrectorAudioProcessor : public juce::AudioProcessor,
                                      public juce::AudioProcessorValueTreeState::Listener
{
public:
    PhaseCorrectorAudioProcessor();
    ~PhaseCorrectorAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    void parameterChanged(const juce::String& parameterID, float newValue) override;

    // Public interface
    bool loadCSVFile(const juce::File& file);
    void setCurvePoints(const std::vector<std::pair<double, double>>& points);
    void clearCurve();
    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts; }
    const std::vector<std::pair<double, double>>& getCurrentCurvePoints() const { return currentCurvePoints; }
    bool hasCurveLoaded() const { return curveLoaded.load(); }

    // Access for visualization
    const NyquistFilter& getNyquistFilter() const { return nyquistFilter; }
    const PhaseProcessor& getPhaseProcessor() const { return phaseProcessor; }

private:
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    juce::AudioProcessorValueTreeState apvts;

    // DSP Components
    PhaseProcessor phaseProcessor;
    OversamplingManager oversamplingManager;
    NyquistFilter nyquistFilter;
    juce::dsp::Gain<float> outputGain;

    // State
    std::vector<std::pair<double, double>> currentCurvePoints;
    std::atomic<bool> curveLoaded{false};

    double lastSampleRate = 44100.0;
    int lastBlockSize = 512;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PhaseCorrectorAudioProcessor)
};
