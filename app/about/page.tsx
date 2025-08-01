import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Heart, Brain, Database, Shield, Users, Activity } from "lucide-react"

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center space-x-2">
              <Heart className="h-8 w-8 text-red-500" />
              <h1 className="text-2xl font-bold text-gray-900">CardioPredict</h1>
            </Link>
            <nav className="flex space-x-4">
              <Link href="/predict" className="text-gray-600 hover:text-gray-900">
                Predict
              </Link>
              <Link href="/about" className="text-gray-900 font-medium">
                About
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">About CardioPredict</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            An AI-powered tool designed to assist healthcare professionals in assessing cardiovascular risk using
            advanced machine learning algorithms.
          </p>
        </div>

        {/* Technology Stack */}
        <Card className="mb-12">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-6 w-6 text-blue-500" />
              <span>Technology & Methodology</span>
            </CardTitle>
            <CardDescription>Built with modern technologies and proven machine learning techniques</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold mb-3">Frontend Technologies</h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>
                    • <strong>Next.js 14:</strong> React framework with App Router
                  </li>
                  <li>
                    • <strong>Tailwind CSS:</strong> Utility-first CSS framework
                  </li>
                  <li>
                    • <strong>TypeScript:</strong> Type-safe development
                  </li>
                  <li>
                    • <strong>shadcn/ui:</strong> Modern component library
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-3">Backend & ML</h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>
                    • <strong>Python:</strong> Core ML development
                  </li>
                  <li>
                    • <strong>FastAPI:</strong> High-performance API framework
                  </li>
                  <li>
                    • <strong>scikit-learn:</strong> Machine learning library
                  </li>
                  <li>
                    • <strong>pandas/numpy:</strong> Data processing
                  </li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Features */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          <Card>
            <CardHeader>
              <Database className="h-8 w-8 text-green-500 mb-2" />
              <CardTitle>Comprehensive Data Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Analyzes 13 key clinical parameters including demographics, vital signs, laboratory results, and
                diagnostic test outcomes to provide accurate risk assessment.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <Activity className="h-8 w-8 text-blue-500 mb-2" />
              <CardTitle>Real-time Predictions</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Instant risk assessment with probability scores and visual feedback to support quick clinical
                decision-making processes.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <Shield className="h-8 w-8 text-purple-500 mb-2" />
              <CardTitle>Privacy & Security</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                No patient data is stored. All processing happens in real-time with secure transmission and immediate
                disposal of sensitive information.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <Users className="h-8 w-8 text-red-500 mb-2" />
              <CardTitle>Healthcare Professional Focus</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Designed specifically for clinical environments with medical terminology, standard units, and workflow
                integration considerations.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Clinical Parameters */}
        <Card className="mb-12">
          <CardHeader>
            <CardTitle>Clinical Parameters Analyzed</CardTitle>
            <CardDescription>The model evaluates the following 13 key indicators</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-3">Demographics & Vitals</h4>
                <ul className="space-y-1 text-sm text-gray-600">
                  <li>• Age and Sex</li>
                  <li>• Resting Blood Pressure</li>
                  <li>• Maximum Heart Rate Achieved</li>
                  <li>• Chest Pain Type Classification</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-3">Laboratory & Diagnostic</h4>
                <ul className="space-y-1 text-sm text-gray-600">
                  <li>• Serum Cholesterol Levels</li>
                  <li>• Fasting Blood Sugar</li>
                  <li>• Resting ECG Results</li>
                  <li>• Exercise-Induced Angina</li>
                  <li>• ST Depression & Slope Analysis</li>
                  <li>• Fluoroscopy Vessel Count</li>
                  <li>• Thalassemia Status</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Model Information */}
        <Card className="mb-12">
          <CardHeader>
            <CardTitle>Machine Learning Model</CardTitle>
            <CardDescription>Technical details about our prediction algorithm</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Training Dataset</h4>
              <p className="text-gray-600 text-sm">
                Trained on the UCI Heart Disease dataset, containing clinical records from multiple medical institutions
                with validated outcomes.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Algorithm</h4>
              <p className="text-gray-600 text-sm">
                Uses ensemble methods including Random Forest and Logistic Regression with cross-validation for optimal
                performance and reliability.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Performance Metrics</h4>
              <p className="text-gray-600 text-sm">
                Optimized for clinical use with high sensitivity and specificity, minimizing both false positives and
                false negatives.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Disclaimer */}
        <Card className="border-yellow-200 bg-yellow-50">
          <CardHeader>
            <CardTitle className="text-yellow-800">Important Medical Disclaimer</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-yellow-700 space-y-2 text-sm">
              <p>
                <strong>This tool is for educational and research purposes only.</strong>
                It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
              </p>
              <p>
                Always seek the advice of qualified healthcare providers with any questions regarding medical
                conditions. Never disregard professional medical advice or delay seeking treatment based on information
                from this tool.
              </p>
              <p>
                The predictions provided are statistical estimates based on population data and may not reflect
                individual patient circumstances or comorbidities.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* CTA */}
        <div className="text-center mt-12">
          <Button asChild size="lg" className="bg-red-500 hover:bg-red-600">
            <Link href="/predict">Try Risk Assessment</Link>
          </Button>
        </div>
      </div>
    </div>
  )
}
