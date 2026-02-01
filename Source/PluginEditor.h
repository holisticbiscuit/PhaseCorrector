/*
  ==============================================================================
    PhaseCorrector - Plugin Editor Header
    Resizable UI with Nyquist filter visualization
  ==============================================================================
*/

#pragma once

#include "PluginProcessor.h"

//==============================================================================
// Nyquist Filter Visualizer
//==============================================================================
class NyquistVisualizer : public juce::Component,
                          public juce::Timer
{
public:
    NyquistVisualizer(PhaseCorrectorAudioProcessor& processor);
    ~NyquistVisualizer() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;

private:
    void updateResponsePath();
    float freqToX(float freq) const;
    float dbToY(float db) const;

    PhaseCorrectorAudioProcessor& audioProcessor;
    juce::Path responsePath;

    static constexpr float MIN_FREQ = 20.0f;
    static constexpr float MAX_FREQ = 40000.0f;
    static constexpr float MIN_DB = -48.0f;
    static constexpr float MAX_DB = 6.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NyquistVisualizer)
};

//==============================================================================
// Phase Curve Display Component with Drawing Support
//==============================================================================
class PhaseCurveDisplay : public juce::Component,
                          public juce::FileDragAndDropTarget,
                          public juce::Timer
{
public:
    PhaseCurveDisplay(PhaseCorrectorAudioProcessor& processor);
    ~PhaseCurveDisplay() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Mouse events for drawing
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    void mouseDoubleClick(const juce::MouseEvent& event) override;

    // File drag and drop
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;
    void fileDragEnter(const juce::StringArray& files, int x, int y) override;
    void fileDragExit(const juce::StringArray& files) override;

    void timerCallback() override;

    // Clear the curve
    void clearCurve();

    // Invert the curve (flip phase signs)
    void invertCurve();

private:
    void drawGrid(juce::Graphics& g);
    void drawCurve(juce::Graphics& g);
    void drawPoints(juce::Graphics& g);
    void drawFrequencyLabels(juce::Graphics& g);
    void drawPhaseLabels(juce::Graphics& g);

    float logFreqToX(float logFreq) const;
    float phaseToY(float normalizedPhase) const;
    float xToLogFreq(float x) const;
    float yToPhase(float y) const;

    void addOrUpdatePoint(float x, float y);
    void updateProcessorCurve();
    int findNearestPointIndex(float x, float y, float maxDistance = 10.0f) const;

    PhaseCorrectorAudioProcessor& audioProcessor;

    // Drawing state
    bool isDrawing = false;
    bool isDraggingPoint = false;
    int draggedPointIndex = -1;
    float lastDrawX = -1.0f;

    // Editable points (stored as logFreq, normalizedPhase pairs)
    std::vector<std::pair<double, double>> editablePoints;

    bool isDraggingOver = false;
    juce::Path curvePath;

    static constexpr float LOG_MIN_FREQ = 1.301030f;
    static constexpr float LOG_MAX_FREQ = 4.301030f;
    static constexpr int MAX_POINTS = 2000;  // High precision like MFreeformPhase
    static constexpr float POINT_SPACING = 0.002f;  // Minimum log freq spacing between points

    int marginLeft = 50;
    int marginRight = 15;
    int marginTop = 15;
    int marginBottom = 30;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PhaseCurveDisplay)
};

//==============================================================================
// Custom Look and Feel
//==============================================================================
class PhaseCorrectorLookAndFeel : public juce::LookAndFeel_V4
{
public:
    PhaseCorrectorLookAndFeel();

    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                          juce::Slider& slider) override;

    void drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown,
                      int buttonX, int buttonY, int buttonW, int buttonH,
                      juce::ComboBox& box) override;

    void drawToggleButton(juce::Graphics& g, juce::ToggleButton& button,
                          bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;

private:
    juce::Colour accentColour;
    juce::Colour backgroundColour;
};

//==============================================================================
// Main Editor Component (Resizable)
//==============================================================================
class PhaseCorrectorAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    explicit PhaseCorrectorAudioProcessorEditor(PhaseCorrectorAudioProcessor&);
    ~PhaseCorrectorAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void setupSlider(juce::Slider& slider, const juce::String& suffix);
    void setupLabel(juce::Label& label, const juce::String& text);

    PhaseCorrectorAudioProcessor& audioProcessor;
    PhaseCorrectorLookAndFeel lookAndFeel;

    // Graph displays
    PhaseCurveDisplay curveDisplay;
    NyquistVisualizer nyquistVisualizer;

    // Main controls
    juce::ComboBox oversampleBox;
    juce::ComboBox ovsModeBox;
    juce::ComboBox fftQualityBox;
    juce::ComboBox fftOverlapBox;
    juce::Slider dryWetSlider;
    juce::Slider outputGainSlider;
    juce::Slider depthSlider;

    // Nyquist controls
    juce::ToggleButton nyquistEnableButton;
    juce::Slider nyquistFreqSlider;
    juce::Slider nyquistQSlider;
    juce::ComboBox nyquistSlopeBox;

    // Buttons
    juce::TextButton clearButton;
    juce::TextButton invertButton;

    // Labels
    juce::Label titleLabel;
    juce::Label oversampleLabel;
    juce::Label ovsModeLabel;
    juce::Label fftQualityLabel;
    juce::Label fftOverlapLabel;
    juce::Label dryWetLabel;
    juce::Label outputGainLabel;
    juce::Label depthLabel;
    juce::Label nyquistLabel;
    juce::Label nyquistFreqLabel;
    juce::Label nyquistQLabel;
    juce::Label nyquistSlopeLabel;
    juce::Label statusLabel;

    // Parameter attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> oversampleAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> ovsModeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> fftQualityAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> fftOverlapAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> dryWetAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> outputGainAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> depthAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> nyquistEnableAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> nyquistFreqAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> nyquistQAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> nyquistSlopeAttachment;

    // Base dimensions for scaling
    static constexpr int BASE_WIDTH = 850;
    static constexpr int BASE_HEIGHT = 650;
    static constexpr int MIN_WIDTH = 640;
    static constexpr int MIN_HEIGHT = 490;
    static constexpr int MAX_WIDTH = 1500;
    static constexpr int MAX_HEIGHT = 1150;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PhaseCorrectorAudioProcessorEditor)
};
