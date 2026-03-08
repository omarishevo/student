const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType, LevelFormat,
  PageNumber, Header, Footer, TabStopType, TabStopPosition
} = require("docx");
const fs = require("fs");
const path = require("path");

const PURPLE = "764ba2";
const BLUE   = "667eea";
const LIGHT  = "F0ECF7";
const WHITE  = "FFFFFF";

// ── Helper: load image ────────────────────────────────────────────────────────
function img(filename, widthEmu, heightEmu) {
  const data = fs.readFileSync(path.join("/home/claude/charts", filename));
  return new ImageRun({ data, transformation: { width: widthEmu / 9144, height: heightEmu / 9144 }, type: "png" });
}

function imgPara(filename, w, h, caption) {
  const items = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 80 },
      children: [img(filename, w, h)]
    })
  ];
  if (caption) {
    items.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: "666666" })]
    }));
  }
  return items;
}

// ── Helper: section heading ───────────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, bold: true, size: 32, color: PURPLE })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 4 } },
    children: [new TextRun({ text, bold: true, size: 26, color: BLUE })]
  });
}
function para(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    children: [new TextRun({ text, size: 22, ...opts })]
  });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 60, after: 60 },
    children: [new TextRun({ text, size: 22 })]
  });
}
function kv(label, value) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    children: [
      new TextRun({ text: `${label}: `, bold: true, size: 22 }),
      new TextRun({ text: value, size: 22 })
    ]
  });
}
function spacer(before = 120) {
  return new Paragraph({ spacing: { before, after: 0 }, children: [] });
}

// ── Stats table ───────────────────────────────────────────────────────────────
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function statsTable() {
  const rows = [
    ["Metric", "Value"],
    ["Total Students", "1,000"],
    ["Average Math Score", "66.1"],
    ["Average Reading Score", "69.2"],
    ["Average Writing Score", "68.1"],
    ["Overall Average Score", "67.8"],
    ["Overall Pass Rate (≥60)", "71.5%"],
    ["Female Students", "51.8%"],
    ["Completed Test Prep", "35.8%"],
  ];
  return new Table({
    width: { size: 6000, type: WidthType.DXA },
    columnWidths: [3000, 3000],
    rows: rows.map((row, i) =>
      new TableRow({
        children: row.map(cell =>
          new TableCell({
            borders,
            width: { size: 3000, type: WidthType.DXA },
            shading: { fill: i === 0 ? PURPLE : (i % 2 === 0 ? LIGHT : WHITE), type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 160, right: 160 },
            children: [new Paragraph({
              children: [new TextRun({
                text: cell,
                bold: i === 0,
                size: i === 0 ? 20 : 20,
                color: i === 0 ? WHITE : "222222"
              })]
            })]
          })
        )
      })
    )
  });
}

// ── File structure table ──────────────────────────────────────────────────────
function fileTable() {
  const rows = [
    ["File", "Description"],
    ["student_performance_app.py", "Main Streamlit application"],
    ["requirements.txt", "Python dependencies"],
    ["StudentsPerformance.csv", "Dataset (upload via sidebar)"],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [4200, 5160],
    rows: rows.map((row, i) =>
      new TableRow({
        children: row.map((cell, j) =>
          new TableCell({
            borders,
            width: { size: j === 0 ? 4200 : 5160, type: WidthType.DXA },
            shading: { fill: i === 0 ? PURPLE : (i % 2 === 0 ? LIGHT : WHITE), type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 160, right: 160 },
            children: [new Paragraph({
              children: [new TextRun({
                text: cell, bold: i === 0, size: 20,
                color: i === 0 ? WHITE : "222222",
                font: j === 0 && i !== 0 ? "Courier New" : undefined
              })]
            })]
          })
        )
      })
    )
  });
}

// ── Document ──────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
      },
      {
        reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
      }
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: PURPLE },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: PURPLE, space: 4 } },
          spacing: { before: 0, after: 120 },
          children: [
            new TextRun({ text: "🎓  Student Performance Predictor", bold: true, size: 22, color: PURPLE }),
            new TextRun({ text: "  ·  README & Visualisation Report", size: 20, color: "888888" })
          ]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 4 } },
          spacing: { before: 80, after: 0 },
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Page ", size: 18, color: "888888" }),
                     new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" }),
                     new TextRun({ text: " of ", size: 18, color: "888888" }),
                     new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "888888" })]
        })]
      })
    },
    children: [
      // ── TITLE BLOCK ──────────────────────────────────────────────────────
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 240, after: 80 },
        children: [new TextRun({ text: "🎓 Student Performance Predictor", bold: true, size: 52, color: PURPLE })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 60 },
        children: [new TextRun({ text: "README & Visualisation Report", size: 28, color: BLUE })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 400 },
        children: [new TextRun({ text: "Built with Python · Streamlit · scikit-learn · Matplotlib · Seaborn", italics: true, size: 20, color: "888888" })]
      }),

      // ── OVERVIEW ─────────────────────────────────────────────────────────
      h1("1. Project Overview"),
      para("This Streamlit application provides an end-to-end machine learning pipeline for exploring and predicting student exam scores. Users can upload the StudentsPerformance.csv dataset and immediately access interactive data exploration, rich visualisations, model training, and real-time prediction — all in a single-page web app."),
      spacer(),
      para("The app supports three ML algorithms (Random Forest, Gradient Boosting, Linear Regression), four score targets (Math, Reading, Writing, Average), and a fully interactive prediction form that returns a grade and dataset percentile for any student profile."),

      // ── DATASET ──────────────────────────────────────────────────────────
      spacer(),
      h1("2. Dataset Summary"),
      para("The dataset contains 1,000 student records across 8 features. Below are the key statistics:"),
      spacer(160),
      statsTable(),
      spacer(200),
      para("Features in the dataset:", { bold: true }),
      bullet("gender — female / male"),
      bullet("race/ethnicity — groups A through E"),
      bullet("parental level of education — 6 levels from some high school to master's degree"),
      bullet("lunch — standard or free/reduced"),
      bullet("test preparation course — completed or none"),
      bullet("math score, reading score, writing score — numeric (0–100)"),

      // ── SETUP ────────────────────────────────────────────────────────────
      spacer(),
      h1("3. Installation & Setup"),
      h2("3.1 Prerequisites"),
      para("Python 3.9 or higher is required."),
      spacer(80),
      h2("3.2 Install Dependencies"),
      new Paragraph({
        spacing: { before: 80, after: 80 },
        shading: { fill: "F4F0FA", type: ShadingType.CLEAR },
        children: [new TextRun({ text: "pip install -r requirements.txt", font: "Courier New", size: 20, color: "333333" })]
      }),
      spacer(80),
      h2("3.3 Run the App"),
      new Paragraph({
        spacing: { before: 80, after: 80 },
        shading: { fill: "F4F0FA", type: ShadingType.CLEAR },
        children: [new TextRun({ text: "streamlit run student_performance_app.py", font: "Courier New", size: 20, color: "333333" })]
      }),
      para("Then open http://localhost:8501 in your browser and upload StudentsPerformance.csv via the sidebar."),

      // ── FILE STRUCTURE ───────────────────────────────────────────────────
      spacer(),
      h1("4. File Structure"),
      spacer(120),
      fileTable(),

      // ── APP FEATURES ─────────────────────────────────────────────────────
      spacer(200),
      h1("5. App Features"),
      h2("Tab 1 — Data Explorer"),
      bullet("Preview the first 20 rows of the uploaded dataset"),
      bullet("Summary statistics for all numeric score columns"),
      bullet("Interactive filters by gender, race/ethnicity, and test preparation status"),
      bullet("Live row count updates as filters are applied"),
      spacer(80),
      h2("Tab 2 — Visualisations"),
      bullet("Score distribution histograms with mean lines"),
      bullet("Grouped bar charts by gender and test preparation"),
      bullet("Correlation heatmap across all score columns"),
      bullet("Average score breakdown by parental education level"),
      spacer(80),
      h2("Tab 3 — ML Model"),
      bullet("Train Random Forest, Gradient Boosting, or Linear Regression"),
      bullet("Configurable test split percentage and tree count via sidebar"),
      bullet("Metrics: RMSE, MAE, R² Score"),
      bullet("Actual vs Predicted scatter plot"),
      bullet("Feature importance bar chart (RF & GB)"),
      spacer(80),
      h2("Tab 4 — Predict"),
      bullet("Input any student profile using dropdown selectors"),
      bullet("Get an instant predicted score, letter grade, and pass/fail status"),
      bullet("See what percentile the predicted score falls in relative to the dataset"),

      // ── VISUALISATIONS ───────────────────────────────────────────────────
      spacer(),
      h1("6. Visualisations"),

      h2("6.1 Score Distributions"),
      para("The histograms below show the spread of Math, Reading, and Writing scores across all 1,000 students. Red dashed lines mark the mean for each subject. Reading and Writing scores are slightly higher on average than Math."),
      ...imgPara("01_distributions.png", 580*914, 155*914, "Figure 1 — Distribution of Math, Reading, and Writing scores"),

      h2("6.2 Average Scores by Gender"),
      para("Female students outperform male students in Reading and Writing, while male students score marginally higher in Math. This pattern is consistent with broad educational research findings."),
      ...imgPara("02_gender.png", 420*914, 230*914, "Figure 2 — Average scores broken down by gender"),

      h2("6.3 Impact of Test Preparation"),
      para("Students who completed the test preparation course score noticeably higher across all three subjects. The improvement is most pronounced in Writing (+7 points on average), suggesting that preparation courses provide targeted writing practice."),
      ...imgPara("03_test_prep.png", 420*914, 230*914, "Figure 3 — Score comparison: completed vs no test preparation"),

      h2("6.4 Score Correlation Heatmap"),
      para("All three scores are strongly correlated with each other (r > 0.80). Reading and Writing share the highest correlation (r ≈ 0.95), indicating that students who perform well in one tend to excel in the other."),
      ...imgPara("04_correlation.png", 340*914, 270*914, "Figure 4 — Pearson correlation matrix for all score columns"),

      h2("6.5 Parental Education vs Average Score"),
      para("There is a clear positive trend between parental education level and student average scores. Students whose parents hold a master's degree average ~6 points higher than those whose parents did not complete high school."),
      ...imgPara("05_parental_edu.png", 530*914, 230*914, "Figure 5 — Average score by parental level of education"),

      h2("6.6 Pass / Fail Rate by Race & Ethnicity"),
      para("Pass rates (average score ≥ 60) vary across racial and ethnic groups, with Group E showing the highest pass rate. The app allows filtering by group to explore these patterns further."),
      ...imgPara("06_pass_fail.png", 480*914, 230*914, "Figure 6 — Pass and fail percentages by race/ethnicity group"),

      // ── ML MODELS ────────────────────────────────────────────────────────
      spacer(),
      h1("7. Machine Learning Models"),
      para("The following algorithms are available in the app:"),
      spacer(80),
      kv("Random Forest", "An ensemble of decision trees trained on random feature subsets. Robust to overfitting and provides feature importance scores. Best default choice for tabular data."),
      spacer(60),
      kv("Gradient Boosting", "Sequentially builds trees to correct previous errors. Often achieves the highest accuracy but is slower to train. Also provides feature importances."),
      spacer(60),
      kv("Linear Regression", "A simple baseline model. Fast and interpretable, but assumes linear relationships. Useful for comparing against ensemble methods."),
      spacer(120),
      para("All models are trained fresh on each run using the settings configured in the sidebar. Categorical features are label-encoded before training."),

      // ── REQUIREMENTS ─────────────────────────────────────────────────────
      spacer(),
      h1("8. Requirements"),
      new Paragraph({
        spacing: { before: 100, after: 100 },
        shading: { fill: "F4F0FA", type: ShadingType.CLEAR },
        children: [
          new TextRun({ text: "streamlit>=1.32.0\n", font: "Courier New", size: 20, color: "333333" }),
          new TextRun({ text: "pandas>=2.0.0\n",     font: "Courier New", size: 20, color: "333333" }),
          new TextRun({ text: "numpy>=1.26.0\n",     font: "Courier New", size: 20, color: "333333" }),
          new TextRun({ text: "matplotlib>=3.8.0\n", font: "Courier New", size: 20, color: "333333" }),
          new TextRun({ text: "seaborn>=0.13.0\n",   font: "Courier New", size: 20, color: "333333" }),
          new TextRun({ text: "scikit-learn>=1.4.0", font: "Courier New", size: 20, color: "333333" }),
        ]
      }),

      // ── NOTES ────────────────────────────────────────────────────────────
      spacer(),
      h1("9. Notes & Limitations"),
      bullet("The app requires the CSV to be uploaded each session — data is not persisted between browser refreshes."),
      bullet("Label encoding is re-applied on each training run; models are not saved to disk."),
      bullet("The dataset contains no missing values, so no imputation is performed."),
      bullet("Average score is derived (mean of the three subjects) and not a raw feature in the dataset."),
      bullet("Predictions use a model retrained on the full dataset for maximum accuracy."),

      spacer(400),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Generated by Claude · Anthropic · 2026", italics: true, size: 18, color: "AAAAAA" })]
      })
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/home/claude/README.docx", buf);
  console.log("Done: README.docx");
});
