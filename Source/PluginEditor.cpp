/*
  ==============================================================================
    PhaseCorrector - Plugin Editor Implementation
    Clean minimal edition - CPU efficient
  ==============================================================================
*/

#include "PluginEditor.h"
#include <cmath>

//==============================================================================
// Simple Color Palette
//==============================================================================
namespace Colors
{
    const juce::Colour background     = juce::Colour(0xff1a1a1a);
    const juce::Colour surface        = juce::Colour(0xff252525);
    const juce::Colour surfaceLight   = juce::Colour(0xff303030);
    const juce::Colour accent         = juce::Colour(0xff4a9eff);
    const juce::Colour accentDim      = juce::Colour(0xff3070b0);
    const juce::Colour text           = juce::Colour(0xffe0e0e0);
    const juce::Colour textDim        = juce::Colour(0xff909090);
    const juce::Colour border         = juce::Colour(0xff404040);
    const juce::Colour graphBg        = juce::Colour(0xff101010);
    const juce::Colour grid           = juce::Colour(0xff2a2a2a);
    const juce::Colour gridCenter     = juce::Colour(0xff404040);
}

//==============================================================================
// Phase Curve Display Implementation
//==============================================================================
PhaseCurveDisplay::PhaseCurveDisplay(PhaseCorrectorAudioProcessor& processor)
    : audioProcessor(processor)
{
    setMouseCursor(juce::MouseCursor::CrosshairCursor);

    // Load existing points from processor
    const auto& processorPoints = audioProcessor.getCurrentCurvePoints();
    if (!processorPoints.empty())
        editablePoints = processorPoints;
}

float PhaseCurveDisplay::logFreqToX(float logFreq) const
{
    const float graphWidth = static_cast<float>(getWidth() - marginLeft - marginRight);
    const float normalized = (logFreq - LOG_MIN_FREQ) / (LOG_MAX_FREQ - LOG_MIN_FREQ);
    return static_cast<float>(marginLeft) + normalized * graphWidth;
}

float PhaseCurveDisplay::phaseToY(float normalizedPhase) const
{
    const float graphHeight = static_cast<float>(getHeight() - marginTop - marginBottom);
    const float normalized = (1.0f - normalizedPhase) / 2.0f;
    return static_cast<float>(marginTop) + normalized * graphHeight;
}

float PhaseCurveDisplay::xToLogFreq(float x) const
{
    const float graphWidth = static_cast<float>(getWidth() - marginLeft - marginRight);
    const float normalized = (x - static_cast<float>(marginLeft)) / graphWidth;
    return LOG_MIN_FREQ + normalized * (LOG_MAX_FREQ - LOG_MIN_FREQ);
}

float PhaseCurveDisplay::yToPhase(float y) const
{
    const float graphHeight = static_cast<float>(getHeight() - marginTop - marginBottom);
    const float normalized = (y - static_cast<float>(marginTop)) / graphHeight;
    return 1.0f - 2.0f * normalized;
}

int PhaseCurveDisplay::findNearestPointIndex(float x, float y, float maxDistance) const
{
    int nearestIdx = -1;
    float nearestDist = maxDistance * maxDistance;

    for (size_t i = 0; i < editablePoints.size(); ++i)
    {
        float px = logFreqToX(static_cast<float>(editablePoints[i].first));
        float py = phaseToY(static_cast<float>(editablePoints[i].second));
        float dist = (px - x) * (px - x) + (py - y) * (py - y);

        if (dist < nearestDist)
        {
            nearestDist = dist;
            nearestIdx = static_cast<int>(i);
        }
    }

    return nearestIdx;
}

void PhaseCurveDisplay::addOrUpdatePoint(float x, float y)
{
    float logFreq = xToLogFreq(x);
    float phase = yToPhase(y);

    logFreq = juce::jlimit(LOG_MIN_FREQ, LOG_MAX_FREQ, logFreq);
    phase = juce::jlimit(-1.0f, 1.0f, phase);

    if (lastDrawX >= 0.0f)
    {
        float lastLogFreq = xToLogFreq(lastDrawX);
        if (std::abs(logFreq - lastLogFreq) < POINT_SPACING)
            return;
    }

    auto it = std::lower_bound(editablePoints.begin(), editablePoints.end(),
                                std::make_pair(static_cast<double>(logFreq), 0.0),
                                [](const auto& a, const auto& b) { return a.first < b.first; });

    if (it != editablePoints.end() && std::abs(it->first - logFreq) < POINT_SPACING)
    {
        it->second = phase;
    }
    else if (it != editablePoints.begin())
    {
        auto prev = std::prev(it);
        if (std::abs(prev->first - logFreq) < POINT_SPACING)
        {
            prev->second = phase;
        }
        else if (editablePoints.size() < MAX_POINTS)
        {
            editablePoints.insert(it, std::make_pair(static_cast<double>(logFreq),
                                                      static_cast<double>(phase)));
        }
    }
    else if (editablePoints.size() < MAX_POINTS)
    {
        editablePoints.insert(it, std::make_pair(static_cast<double>(logFreq),
                                                  static_cast<double>(phase)));
    }

    lastDrawX = x;
}

void PhaseCurveDisplay::updateProcessorCurve()
{
    std::sort(editablePoints.begin(), editablePoints.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    audioProcessor.setCurvePoints(editablePoints);
}

void PhaseCurveDisplay::clearCurve()
{
    editablePoints.clear();
    audioProcessor.clearCurve();
    repaint();
}

void PhaseCurveDisplay::loadFromProcessor()
{
    const auto& processorPoints = audioProcessor.getCurrentCurvePoints();
    editablePoints.clear();
    editablePoints.reserve(processorPoints.size());
    for (const auto& pt : processorPoints)
        editablePoints.push_back(pt);
    repaint();
}

void PhaseCurveDisplay::invertCurve()
{
    for (auto& point : editablePoints)
        point.second = -point.second;
    updateProcessorCurve();
    repaint();
}

void PhaseCurveDisplay::mouseDown(const juce::MouseEvent& event)
{
    if (event.mods.isLeftButtonDown())
    {
        float x = static_cast<float>(event.x);
        float y = static_cast<float>(event.y);

        if (x >= marginLeft && x <= getWidth() - marginRight &&
            y >= marginTop && y <= getHeight() - marginBottom)
        {
            draggedPointIndex = findNearestPointIndex(x, y, 8.0f);

            if (draggedPointIndex >= 0)
            {
                isDraggingPoint = true;
            }
            else
            {
                isDrawing = true;
                lastDrawX = -1.0f;
                addOrUpdatePoint(x, y);
            }
        }
    }
    else if (event.mods.isRightButtonDown())
    {
        float x = static_cast<float>(event.x);
        float y = static_cast<float>(event.y);
        int idx = findNearestPointIndex(x, y, 15.0f);

        if (idx >= 0)
        {
            editablePoints.erase(editablePoints.begin() + idx);
            updateProcessorCurve();
        }
    }
}

void PhaseCurveDisplay::mouseDrag(const juce::MouseEvent& event)
{
    float x = juce::jlimit(static_cast<float>(marginLeft),
                           static_cast<float>(getWidth() - marginRight),
                           static_cast<float>(event.x));
    float y = juce::jlimit(static_cast<float>(marginTop),
                           static_cast<float>(getHeight() - marginBottom),
                           static_cast<float>(event.y));

    if (isDrawing)
    {
        if (lastDrawX >= 0.0f)
        {
            float dx = x - lastDrawX;
            int steps = std::max(1, static_cast<int>(std::abs(dx) / 2.0f));

            for (int i = 1; i <= steps; ++i)
            {
                float t = static_cast<float>(i) / steps;
                float interpX = lastDrawX + dx * t;
                float interpY = static_cast<float>(event.y - event.getDistanceFromDragStartY()) +
                               (y - (event.y - event.getDistanceFromDragStartY())) * t;
                interpY = juce::jlimit(static_cast<float>(marginTop),
                                        static_cast<float>(getHeight() - marginBottom), interpY);
                addOrUpdatePoint(interpX, interpY);
            }
        }
        else
        {
            addOrUpdatePoint(x, y);
        }
        lastDrawX = x;
    }
    else if (isDraggingPoint && draggedPointIndex >= 0)
    {
        float logFreq = xToLogFreq(x);
        float phase = yToPhase(y);

        logFreq = juce::jlimit(LOG_MIN_FREQ, LOG_MAX_FREQ, logFreq);
        phase = juce::jlimit(-1.0f, 1.0f, phase);

        editablePoints[draggedPointIndex].first = logFreq;
        editablePoints[draggedPointIndex].second = phase;
    }

    repaint();
}

void PhaseCurveDisplay::mouseUp(const juce::MouseEvent&)
{
    if (isDrawing || isDraggingPoint)
    {
        updateProcessorCurve();
    }

    isDrawing = false;
    isDraggingPoint = false;
    draggedPointIndex = -1;
    lastDrawX = -1.0f;
}

void PhaseCurveDisplay::mouseDoubleClick(const juce::MouseEvent& event)
{
    float x = static_cast<float>(event.x);
    float y = static_cast<float>(event.y);

    if (x >= marginLeft && x <= getWidth() - marginRight &&
        y >= marginTop && y <= getHeight() - marginBottom)
    {
        float logFreq = xToLogFreq(x);
        float phase = yToPhase(y);

        logFreq = juce::jlimit(LOG_MIN_FREQ, LOG_MAX_FREQ, logFreq);
        phase = juce::jlimit(-1.0f, 1.0f, phase);

        editablePoints.push_back(std::make_pair(static_cast<double>(logFreq),
                                                 static_cast<double>(phase)));
        updateProcessorCurve();
        repaint();
    }
}

void PhaseCurveDisplay::drawGrid(juce::Graphics& g)
{
    auto graphArea = getLocalBounds()
        .withTrimmedLeft(marginLeft)
        .withTrimmedRight(marginRight)
        .withTrimmedTop(marginTop)
        .withTrimmedBottom(marginBottom);

    // Graph background - single flat fill
    g.setColour(Colors::graphBg);
    g.fillRoundedRectangle(graphArea.toFloat(), 4.0f);

    // Frequency grid lines
    std::array<float, 9> freqs = { 20.0f, 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f, 2000.0f, 5000.0f, 10000.0f };
    g.setColour(Colors::grid);
    for (float freq : freqs)
    {
        float x = logFreqToX(std::log10(freq));
        g.drawVerticalLine(static_cast<int>(x), static_cast<float>(marginTop), static_cast<float>(getHeight() - marginBottom));
    }

    // Phase grid lines
    std::array<float, 5> phases = { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f };
    for (float phase : phases)
    {
        float y = phaseToY(phase);
        g.setColour(std::abs(phase) < 0.01f ? Colors::gridCenter : Colors::grid);
        g.drawHorizontalLine(static_cast<int>(y), static_cast<float>(marginLeft), static_cast<float>(getWidth() - marginRight));
    }

    // Simple border
    g.setColour(Colors::border);
    g.drawRoundedRectangle(graphArea.toFloat(), 4.0f, 1.0f);
}

void PhaseCurveDisplay::drawFrequencyLabels(juce::Graphics& g)
{
    g.setColour(Colors::textDim);
    g.setFont(juce::FontOptions(static_cast<float>(getWidth()) / 70.0f));

    struct FreqLabel { float freq; const char* text; };
    std::array<FreqLabel, 7> labels = {{
        { 20.0f, "20" }, { 100.0f, "100" }, { 500.0f, "500" },
        { 1000.0f, "1k" }, { 5000.0f, "5k" }, { 10000.0f, "10k" }, { 20000.0f, "20k" }
    }};

    for (const auto& label : labels)
    {
        float x = logFreqToX(std::log10(label.freq));
        g.drawText(label.text, static_cast<int>(x) - 20, getHeight() - marginBottom + 5,
                   40, 20, juce::Justification::centredTop);
    }
}

void PhaseCurveDisplay::drawPhaseLabels(juce::Graphics& g)
{
    g.setColour(Colors::textDim);
    g.setFont(juce::FontOptions(static_cast<float>(getWidth()) / 70.0f));

    struct PhaseLabel { float phase; const char* text; };
    std::array<PhaseLabel, 5> labels = {{
        { 1.0f, "+360" }, { 0.5f, "+180" }, { 0.0f, "0" }, { -0.5f, "-180" }, { -1.0f, "-360" }
    }};

    for (const auto& label : labels)
    {
        float y = phaseToY(label.phase);
        g.setColour(std::abs(label.phase) < 0.01f ? Colors::text : Colors::textDim);
        g.drawText(label.text, 5, static_cast<int>(y) - 8, marginLeft - 10, 16,
                   juce::Justification::centredRight);
    }
}

void PhaseCurveDisplay::drawCurve(juce::Graphics& g)
{
    const auto& points = editablePoints.empty() ? audioProcessor.getCurrentCurvePoints() : editablePoints;

    if (points.empty())
    {
        g.setColour(Colors::textDim);
        g.setFont(juce::FontOptions(static_cast<float>(getWidth()) / 50.0f));
        g.drawText("Draw your phase curve", getLocalBounds(), juce::Justification::centred);
        return;
    }

    // Build path
    juce::Path curvePath;
    bool started = false;

    for (const auto& point : points)
    {
        float x = logFreqToX(static_cast<float>(point.first));
        float y = phaseToY(static_cast<float>(point.second));

        if (!started)
        {
            curvePath.startNewSubPath(x, y);
            started = true;
        }
        else
        {
            curvePath.lineTo(x, y);
        }
    }

    // Single stroke for the curve (CPU efficient)
    g.setColour(Colors::accent);
    g.strokePath(curvePath, juce::PathStrokeType(2.0f, juce::PathStrokeType::curved));

    // Point count indicator
    g.setColour(Colors::textDim);
    g.setFont(juce::FontOptions(10.0f));
    g.drawText(juce::String(points.size()) + " pts",
               getWidth() - marginRight - 50, marginTop + 5, 45, 15,
               juce::Justification::right);
}

void PhaseCurveDisplay::paint(juce::Graphics& g)
{
    drawGrid(g);
    drawCurve(g);
    drawFrequencyLabels(g);
    drawPhaseLabels(g);

    if (isDraggingOver)
    {
        auto graphArea = getLocalBounds()
            .withTrimmedLeft(marginLeft).withTrimmedRight(marginRight)
            .withTrimmedTop(marginTop).withTrimmedBottom(marginBottom);

        g.setColour(Colors::accent.withAlpha(0.2f));
        g.fillRoundedRectangle(graphArea.toFloat(), 4.0f);

        g.setColour(Colors::accent);
        g.drawRoundedRectangle(graphArea.toFloat(), 4.0f, 2.0f);

        g.setColour(Colors::text);
        g.setFont(juce::FontOptions(16.0f));
        g.drawText("DROP CSV FILE", getLocalBounds(), juce::Justification::centred);
    }

    if (isDrawing)
    {
        g.setColour(Colors::accent);
        g.setFont(juce::FontOptions(10.0f));
        g.drawText("DRAWING", marginLeft + 5, marginTop + 5, 60, 15, juce::Justification::left);
    }
}

void PhaseCurveDisplay::resized()
{
    float scale = static_cast<float>(getWidth()) / 600.0f;
    marginLeft = static_cast<int>(50 * scale);
    marginRight = static_cast<int>(15 * scale);
    marginTop = static_cast<int>(15 * scale);
    marginBottom = static_cast<int>(30 * scale);
}

bool PhaseCurveDisplay::isInterestedInFileDrag(const juce::StringArray& files)
{
    for (const auto& file : files)
        if (file.endsWithIgnoreCase(".csv") || file.endsWithIgnoreCase(".txt"))
            return true;
    return false;
}

void PhaseCurveDisplay::filesDropped(const juce::StringArray& files, int, int)
{
    isDraggingOver = false;

    for (const auto& filePath : files)
    {
        juce::File file(filePath);
        if (file.existsAsFile() &&
            (file.hasFileExtension(".csv") || file.hasFileExtension(".txt")))
        {
            if (audioProcessor.loadCSVFile(file))
            {
                editablePoints = audioProcessor.getCurrentCurvePoints();
                repaint();
                break;
            }
        }
    }
}

void PhaseCurveDisplay::fileDragEnter(const juce::StringArray&, int, int)
{
    isDraggingOver = true;
    repaint();
}

void PhaseCurveDisplay::fileDragExit(const juce::StringArray&)
{
    isDraggingOver = false;
    repaint();
}

//==============================================================================
// Simple Look and Feel
//==============================================================================
PhaseCorrectorLookAndFeel::PhaseCorrectorLookAndFeel()
{
    setColour(juce::Slider::rotarySliderFillColourId, Colors::accent);
    setColour(juce::Slider::rotarySliderOutlineColourId, Colors::border);
    setColour(juce::ComboBox::backgroundColourId, Colors::surface);
    setColour(juce::ComboBox::textColourId, Colors::text);
    setColour(juce::ComboBox::outlineColourId, Colors::border);
    setColour(juce::ComboBox::arrowColourId, Colors::accent);
    setColour(juce::PopupMenu::backgroundColourId, Colors::surface);
    setColour(juce::PopupMenu::textColourId, Colors::text);
    setColour(juce::PopupMenu::highlightedBackgroundColourId, Colors::surfaceLight);
    setColour(juce::Label::textColourId, Colors::text);
    setColour(juce::TextButton::buttonColourId, Colors::surface);
    setColour(juce::TextButton::textColourOffId, Colors::text);
}

void PhaseCorrectorLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                                  float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                                                  juce::Slider& slider)
{
    const float radius = static_cast<float>(juce::jmin(width, height)) / 2.0f - 4.0f;
    const float centreX = static_cast<float>(x) + static_cast<float>(width) * 0.5f;
    const float centreY = static_cast<float>(y) + static_cast<float>(height) * 0.5f;
    const float angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);

    // Background track
    juce::Path backgroundArc;
    backgroundArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                                 rotaryStartAngle, rotaryEndAngle, true);
    g.setColour(Colors::surface);
    g.strokePath(backgroundArc, juce::PathStrokeType(5.0f, juce::PathStrokeType::curved,
                                                      juce::PathStrokeType::rounded));

    // Value arc
    juce::Path valueArc;
    valueArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                           rotaryStartAngle, angle, true);
    g.setColour(Colors::accent);
    g.strokePath(valueArc, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));

    // Center knob
    float knobRadius = radius * 0.5f;
    g.setColour(Colors::surfaceLight);
    g.fillEllipse(centreX - knobRadius, centreY - knobRadius, knobRadius * 2, knobRadius * 2);
    g.setColour(Colors::border);
    g.drawEllipse(centreX - knobRadius, centreY - knobRadius, knobRadius * 2, knobRadius * 2, 1.0f);

    // Pointer
    juce::Path pointer;
    const float pointerLength = radius * 0.5f;
    const float pointerThickness = 2.0f;
    pointer.addRoundedRectangle(-pointerThickness * 0.5f, -radius + 6.0f,
                                 pointerThickness, pointerLength, 1.0f);

    g.setColour(Colors::text);
    g.fillPath(pointer, juce::AffineTransform::rotation(angle).translated(centreX, centreY));

    // Value text
    g.setColour(Colors::text);
    g.setFont(juce::FontOptions(std::max(9.0f, static_cast<float>(width) / 7.5f)));
    juce::String valueText = slider.getTextFromValue(slider.getValue());
    g.drawText(valueText, x, static_cast<int>(centreY + radius * 0.6f),
               width, 15, juce::Justification::centred, false);
}

void PhaseCorrectorLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool,
                                              int, int, int, int, juce::ComboBox&)
{
    // Simple flat combo box
    g.setColour(Colors::surface);
    g.fillRoundedRectangle(0, 0, static_cast<float>(width), static_cast<float>(height), 4.0f);

    g.setColour(Colors::border);
    g.drawRoundedRectangle(0.5f, 0.5f, static_cast<float>(width) - 1.0f,
                            static_cast<float>(height) - 1.0f, 4.0f, 1.0f);

    // Arrow
    juce::Rectangle<int> arrowZone(width - 22, 0, 18, height);
    juce::Path path;
    path.startNewSubPath(static_cast<float>(arrowZone.getX()) + 3.0f,
                         static_cast<float>(arrowZone.getCentreY()) - 2.0f);
    path.lineTo(static_cast<float>(arrowZone.getCentreX()),
                static_cast<float>(arrowZone.getCentreY()) + 3.0f);
    path.lineTo(static_cast<float>(arrowZone.getRight()) - 3.0f,
                static_cast<float>(arrowZone.getCentreY()) - 2.0f);

    g.setColour(Colors::accent);
    g.strokePath(path, juce::PathStrokeType(1.5f));
}

void PhaseCorrectorLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button,
                                                      const juce::Colour&,
                                                      bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown)
{
    auto bounds = button.getLocalBounds().toFloat().reduced(1.0f);

    if (shouldDrawButtonAsDown)
        g.setColour(Colors::accentDim);
    else if (shouldDrawButtonAsHighlighted)
        g.setColour(Colors::surfaceLight);
    else
        g.setColour(Colors::surface);

    g.fillRoundedRectangle(bounds, 4.0f);

    g.setColour(shouldDrawButtonAsDown ? Colors::accent : Colors::border);
    g.drawRoundedRectangle(bounds, 4.0f, 1.0f);
}

void PhaseCorrectorLookAndFeel::drawButtonText(juce::Graphics& g, juce::TextButton& button,
                                                bool, bool shouldDrawButtonAsDown)
{
    auto bounds = button.getLocalBounds().toFloat();
    g.setColour(shouldDrawButtonAsDown ? Colors::text : Colors::text);
    g.setFont(juce::FontOptions(std::max(10.0f, bounds.getHeight() * 0.4f)));
    g.drawText(button.getButtonText(), bounds.toNearestInt(), juce::Justification::centred);
}

//==============================================================================
// Main Editor Implementation
//==============================================================================
PhaseCorrectorAudioProcessorEditor::PhaseCorrectorAudioProcessorEditor(PhaseCorrectorAudioProcessor& p)
    : AudioProcessorEditor(&p),
      audioProcessor(p),
      curveDisplay(p)
{
    setLookAndFeel(&lookAndFeel);

    setResizable(true, true);
    setResizeLimits(MIN_WIDTH, MIN_HEIGHT, MAX_WIDTH, MAX_HEIGHT);
    getConstrainer()->setFixedAspectRatio(static_cast<double>(BASE_WIDTH) / BASE_HEIGHT);

    // Title
    titleLabel.setText("Phase Corrector", juce::dontSendNotification);
    titleLabel.setFont(juce::FontOptions(20.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, Colors::text);
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(titleLabel);

    addAndMakeVisible(curveDisplay);

    // Clear button
    clearButton.setButtonText("Clear");
    clearButton.onClick = [this]() { curveDisplay.clearCurve(); };
    addAndMakeVisible(clearButton);

    invertButton.setButtonText("Invert");
    invertButton.onClick = [this]() { curveDisplay.invertCurve(); };
    addAndMakeVisible(invertButton);

    // Preset controls
    presetBox.setTextWhenNothingSelected("Presets...");
    presetBox.onChange = [this]()
    {
        if (presetBox.getSelectedId() > 0)
        {
            auto presetName = presetBox.getText();
            audioProcessor.getPresetManager().loadPreset(presetName);
            curveDisplay.loadFromProcessor();
        }
    };
    addAndMakeVisible(presetBox);
    refreshPresetList();

    savePresetButton.setButtonText("Save");
    savePresetButton.onClick = [this]()
    {
        auto* alertWindow = new juce::AlertWindow("Save Preset", "Enter preset name:", juce::MessageBoxIconType::NoIcon);
        auto currentName = presetBox.getText();
        if (currentName == "Presets..." || currentName.isEmpty())
            currentName = "";
        alertWindow->addTextEditor("presetName", currentName, "Name:");
        alertWindow->addButton("Save", 1, juce::KeyPress(juce::KeyPress::returnKey));
        alertWindow->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));

        alertWindow->enterModalState(true, juce::ModalCallbackFunction::create(
            [this, alertWindow](int result)
            {
                if (result == 1)
                {
                    auto presetName = alertWindow->getTextEditorContents("presetName");
                    if (presetName.isNotEmpty())
                    {
                        audioProcessor.getPresetManager().savePreset(presetName);
                        refreshPresetList();
                    }
                }
                delete alertWindow;
            }), true);
    };
    addAndMakeVisible(savePresetButton);

    deletePresetButton.setButtonText("Del");
    deletePresetButton.onClick = [this]()
    {
        auto name = presetBox.getText();
        if (name.isNotEmpty() && name != "Presets...")
        {
            audioProcessor.getPresetManager().deletePreset(name);
            refreshPresetList();
        }
    };
    addAndMakeVisible(deletePresetButton);

    // Main controls
    setupLabel(fftQualityLabel, "Quality");
    fftQualityBox.addItemList(PhaseProcessor::getQualityNames(), 1);
    addAndMakeVisible(fftQualityBox);
    fftQualityAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getAPVTS(), "fftQuality", fftQualityBox);

    setupLabel(fftOverlapLabel, "Overlap");
    fftOverlapBox.addItemList(PhaseProcessor::getOverlapNames(), 1);
    addAndMakeVisible(fftOverlapBox);
    fftOverlapAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getAPVTS(), "fftOverlap", fftOverlapBox);

    setupLabel(depthLabel, "Depth");
    setupSlider(depthSlider, "%");
    depthAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "depth", depthSlider);

    setupLabel(dryWetLabel, "Mix");
    setupSlider(dryWetSlider, "%");
    dryWetAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "dryWet", dryWetSlider);

    setupLabel(outputGainLabel, "Output");
    setupSlider(outputGainSlider, "dB");
    outputGainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "outputGain", outputGainSlider);

    setSize(BASE_WIDTH, BASE_HEIGHT);
}

PhaseCorrectorAudioProcessorEditor::~PhaseCorrectorAudioProcessorEditor()
{
    setLookAndFeel(nullptr);
}

void PhaseCorrectorAudioProcessorEditor::setupSlider(juce::Slider& slider, const juce::String& suffix)
{
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    slider.setTextValueSuffix(suffix);
    addAndMakeVisible(slider);
}

void PhaseCorrectorAudioProcessorEditor::setupLabel(juce::Label& label, const juce::String& text)
{
    label.setText(text, juce::dontSendNotification);
    label.setFont(juce::FontOptions(11.0f));
    label.setColour(juce::Label::textColourId, Colors::textDim);
    label.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(label);
}

void PhaseCorrectorAudioProcessorEditor::refreshPresetList()
{
    auto currentText = presetBox.getText();
    presetBox.clear();

    auto presets = audioProcessor.getPresetManager().getPresetList();
    int id = 1;
    int selectedId = 0;

    for (const auto& preset : presets)
    {
        presetBox.addItem(preset, id);
        if (preset == currentText)
            selectedId = id;
        id++;
    }

    if (selectedId > 0)
        presetBox.setSelectedId(selectedId, juce::dontSendNotification);
}

void PhaseCorrectorAudioProcessorEditor::paint(juce::Graphics& g)
{
    // Simple flat background
    g.fillAll(Colors::background);

    // Control panel separator line
    float scale = static_cast<float>(getWidth()) / BASE_WIDTH;
    int controlsY = static_cast<int>(320 * scale);
    g.setColour(Colors::border);
    g.drawHorizontalLine(controlsY, 0.0f, static_cast<float>(getWidth()));
}

void PhaseCorrectorAudioProcessorEditor::resized()
{
    float scale = static_cast<float>(getWidth()) / BASE_WIDTH;

    // Header
    int headerH = static_cast<int>(40 * scale);
    titleLabel.setBounds(static_cast<int>(15 * scale), 0,
                         static_cast<int>(150 * scale), headerH);

    // Clear and Invert buttons
    clearButton.setBounds(static_cast<int>(170 * scale), static_cast<int>(8 * scale),
                          static_cast<int>(50 * scale), static_cast<int>(24 * scale));
    invertButton.setBounds(static_cast<int>(225 * scale), static_cast<int>(8 * scale),
                           static_cast<int>(50 * scale), static_cast<int>(24 * scale));

    // Preset controls
    presetBox.setBounds(static_cast<int>(290 * scale), static_cast<int>(8 * scale),
                        static_cast<int>(120 * scale), static_cast<int>(24 * scale));
    savePresetButton.setBounds(static_cast<int>(415 * scale), static_cast<int>(8 * scale),
                               static_cast<int>(45 * scale), static_cast<int>(24 * scale));
    deletePresetButton.setBounds(static_cast<int>(465 * scale), static_cast<int>(8 * scale),
                                 static_cast<int>(35 * scale), static_cast<int>(24 * scale));

    // Phase curve display
    int curveY = headerH;
    int curveH = static_cast<int>(270 * scale);
    curveDisplay.setBounds(static_cast<int>(10 * scale), curveY,
                           getWidth() - static_cast<int>(20 * scale), curveH);

    // Controls area
    int controlsY = curveY + curveH + static_cast<int>(15 * scale);
    int knobSize = static_cast<int>(65 * scale);
    int labelH = static_cast<int>(14 * scale);
    int spacing = static_cast<int>(20 * scale);

    int totalControlsWidth = static_cast<int>(90 * scale) * 2 + static_cast<int>(10 * scale) +
                              knobSize * 3 + spacing * 2;
    int startX = (getWidth() - totalControlsWidth) / 2;

    int x = startX;
    int y = controlsY + static_cast<int>(5 * scale);
    int comboW = static_cast<int>(90 * scale);
    int comboH = static_cast<int>(24 * scale);
    int comboSpacing = static_cast<int>(10 * scale);

    // FFT Quality
    fftQualityLabel.setBounds(x, y, comboW, labelH);
    fftQualityBox.setBounds(x, y + labelH + 2, comboW, comboH);

    x += comboW + comboSpacing;

    // Overlap
    fftOverlapLabel.setBounds(x, y, comboW, labelH);
    fftOverlapBox.setBounds(x, y + labelH + 2, comboW, comboH);

    x += comboW + spacing;

    // Depth
    depthLabel.setBounds(x, y, knobSize, labelH);
    depthSlider.setBounds(x, y + labelH, knobSize, knobSize);

    x += knobSize + spacing;

    // Dry/Wet
    dryWetLabel.setBounds(x, y, knobSize, labelH);
    dryWetSlider.setBounds(x, y + labelH, knobSize, knobSize);

    x += knobSize + spacing;

    // Output
    outputGainLabel.setBounds(x, y, knobSize, labelH);
    outputGainSlider.setBounds(x, y + labelH, knobSize, knobSize);
}
