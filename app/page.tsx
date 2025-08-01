import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Heart, Activity, Users, Shield } from "lucide-react"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Heart className="h-8 w-8 text-red-500" />
              <h1 className="text-2xl font-bold text-gray-900">CardioPredict</h1>
            </div>
            <nav className="flex space-x-4">
              <Link href="/predict" className="text-gray-600 hover:text-gray-900">
                Predict
              </Link>
              <Link href="/about" className="text-gray-600 hover:text-gray-900">
                About
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">AI-Powered Heart Disease Risk Assessment</h2>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Leverage advanced machine learning to assess cardiovascular risk based on clinical parameters. Get instant
            predictions to support healthcare decision-making.
          </p>
          <div className="flex justify-center space-x-4">
            <Button asChild size="lg" className="bg-red-500 hover:bg-red-600">
              <Link href="/predict">Start Assessment</Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/about">Learn More</Link>
            </Button>
          </div>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <Card>
            <CardHeader>
              <Activity className="h-12 w-12 text-blue-500 mb-4" />
              <CardTitle>Advanced ML Model</CardTitle>
              <CardDescription>
                Trained on comprehensive heart disease datasets using state-of-the-art algorithms
              </CardDescription>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <Users className="h-12 w-12 text-green-500 mb-4" />
              <CardTitle>Healthcare Professional Ready</CardTitle>
              <CardDescription>
                Designed for clinical environments with medical-grade accuracy and reliability
              </CardDescription>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <Shield className="h-12 w-12 text-purple-500 mb-4" />
              <CardTitle>Secure & Private</CardTitle>
              <CardDescription>
                Patient data is processed securely with no storage of personal information
              </CardDescription>
            </CardHeader>
          </Card>
        </div>

        {/* How it Works */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h3 className="text-2xl font-bold text-center mb-8">How It Works</h3>
          <div className="grid md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-blue-600">1</span>
              </div>
              <h4 className="font-semibold mb-2">Input Patient Data</h4>
              <p className="text-gray-600 text-sm">Enter clinical parameters and health metrics</p>
            </div>
            <div className="text-center">
              <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-green-600">2</span>
              </div>
              <h4 className="font-semibold mb-2">AI Analysis</h4>
              <p className="text-gray-600 text-sm">Machine learning model processes the data</p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-purple-600">3</span>
              </div>
              <h4 className="font-semibold mb-2">Risk Assessment</h4>
              <p className="text-gray-600 text-sm">Get instant risk prediction results</p>
            </div>
            <div className="text-center">
              <div className="bg-red-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-red-600">4</span>
              </div>
              <h4 className="font-semibold mb-2">Clinical Decision</h4>
              <p className="text-gray-600 text-sm">Use results to inform treatment decisions</p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p>&copy; 2024 CardioPredict. Built for healthcare professionals.</p>
          <p className="text-gray-400 text-sm mt-2">
            This tool is for educational and research purposes. Always consult healthcare professionals for medical
            decisions.
          </p>
        </div>
      </footer>
    </div>
  )
}
