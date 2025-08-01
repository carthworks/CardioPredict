# Heart Disease Risk Prediction Tool

An AI-powered web application that combines machine learning with modern web technologies to assess cardiovascular risk based on clinical parameters.

## 🎯 Overview

This application provides healthcare professionals and researchers with an intelligent tool for heart disease risk assessment. It uses a trained machine learning model to analyze patient clinical data and provide instant risk predictions.

## 🛠 Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Modern component library
- **Lucide React** - Beautiful icons

### Backend & ML
- **Python 3.8+** - Core ML development
- **FastAPI** - High-performance API framework
- **scikit-learn** - Machine learning library
- **pandas & numpy** - Data processing
- **joblib** - Model serialization

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.8+
- pip (Python package manager)

### Frontend Setup

1. **Install dependencies:**
   \`\`\`bash
   npm install
   \`\`\`

2. **Run development server:**
   \`\`\`bash
   npm run dev
   \`\`\`

3. **Open your browser:**
   Navigate to `http://localhost:3000`

### Backend Setup

1. **Create virtual environment:**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

2. **Install Python dependencies:**
   \`\`\`bash
   pip install -r scripts/requirements.txt
   \`\`\`

3. **Train the ML model:**
   \`\`\`bash
   cd scripts
   python train_model.py
   \`\`\`

4. **Start the API server:**
   \`\`\`bash
   python api_server.py
   \`\`\`

The API will be available at `http://localhost:8000`

## 📊 Features

### 🏥 Clinical Assessment
- Comprehensive 13-parameter evaluation
- Real-time risk prediction
- Visual risk indicators
- Probability scoring

### 🤖 AI/ML Capabilities
- Trained on heart disease datasets
- Multiple algorithm comparison
- Cross-validation and hyperparameter tuning
- Feature importance analysis

### 🎨 User Experience
- Responsive design
- Intuitive form interface
- Clear result visualization
- Medical terminology support

### 🔒 Security & Privacy
- No data storage
- Secure API communication
- Input validation
- Error handling

## 📋 Clinical Parameters

The model analyzes these key indicators:

**Demographics & Vitals:**
- Age and Sex
- Resting Blood Pressure
- Maximum Heart Rate Achieved
- Chest Pain Type Classification

**Laboratory & Diagnostic:**
- Serum Cholesterol Levels
- Fasting Blood Sugar
- Resting ECG Results
- Exercise-Induced Angina
- ST Depression & Slope Analysis
- Fluoroscopy Vessel Count
- Thalassemia Status

## 🔧 API Endpoints

### `POST /predict`
Predict heart disease risk for a single patient.

**Request Body:**
\`\`\`json
{
  "age": 45,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.5,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
\`\`\`

**Response:**
\`\`\`json
{
  "prediction": 1,
  "probability": 0.75,
  "risk_level": "High Risk",
  "confidence": 0.85
}
\`\`\`

### `GET /health`
Check API health status.

### `GET /model-info`
Get information about the loaded model.

## 🧪 Testing

### Frontend Testing
\`\`\`bash
npm run lint
npm run type-check
\`\`\`

### Backend Testing
\`\`\`bash
cd scripts
python -m pytest test_api.py -v
\`\`\`

### Manual API Testing
\`\`\`bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }'
\`\`\`

## 📁 Project Structure

\`\`\`
heart-disease-predictor/
├── app/                          # Next.js app directory
│   ├── page.tsx                  # Landing page
│   ├── predict/
│   │   └── page.tsx             # Prediction form
│   ├── about/
│   │   └── page.tsx             # About page
│   ├── layout.tsx               # Root layout
│   └── globals.css              # Global styles
├── components/
│   └── ui/                      # shadcn/ui components
├── scripts/                     # Python backend
│   ├── train_model.py          # ML model training
│   ├── api_server.py           # FastAPI server
│   ├── requirements.txt        # Python dependencies
│   └── test_api.py             # API tests
├── public/                     # Static assets
├── README.md                   # Project documentation
├── package.json               # Node.js dependencies
├── tailwind.config.ts         # Tailwind configuration
└── tsconfig.json             # TypeScript configuration
\`\`\`

## 🚀 Deployment

### Frontend (Vercel)
1. Push code to GitHub
2. Connect repository to Vercel
3. Deploy automatically

### Backend (Railway/Render)
1. Create `Dockerfile`:
\`\`\`dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY scripts/requirements.txt .
RUN pip install -r requirements.txt

COPY scripts/ .
EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

2. Deploy to Railway or Render
3. Update frontend API endpoints

## ⚠️ Important Disclaimers

**Medical Disclaimer:** This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

**Data Privacy:** No patient data is stored by this application. All processing happens in real-time with immediate disposal of sensitive information.

**Accuracy:** While the model is trained on validated datasets, predictions are statistical estimates and may not reflect individual patient circumstances.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for heart disease datasets
- Healthcare professionals who provided domain expertise
- Open source community for tools and libraries

## 📞 Support

For questions or support:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

---

**Built with ❤️ for healthcare innovation**
