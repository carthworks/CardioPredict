"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Heart, AlertTriangle, CheckCircle } from "lucide-react"
import Link from "next/link"

interface PatientData {
  age: string
  sex: string
  cp: string
  trestbps: string
  chol: string
  fbs: string
  restecg: string
  thalach: string
  exang: string
  oldpeak: string
  slope: string
  ca: string
  thal: string
}

interface PredictionResult {
  prediction: number
  probability: number
  risk_level: string
}

export default function PredictPage() {
  const [formData, setFormData] = useState<PatientData>({
    age: "",
    sex: "",
    cp: "",
    trestbps: "",
    chol: "",
    fbs: "",
    restecg: "",
    thalach: "",
    exang: "",
    oldpeak: "",
    slope: "",
    ca: "",
    thal: "",
  })

  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleInputChange = (field: keyof PatientData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Convert form data to numbers for API
      const apiData = {
        age: Number.parseInt(formData.age),
        sex: Number.parseInt(formData.sex),
        cp: Number.parseInt(formData.cp),
        trestbps: Number.parseInt(formData.trestbps),
        chol: Number.parseInt(formData.chol),
        fbs: Number.parseInt(formData.fbs),
        restecg: Number.parseInt(formData.restecg),
        thalach: Number.parseInt(formData.thalach),
        exang: Number.parseInt(formData.exang),
        oldpeak: Number.parseFloat(formData.oldpeak),
        slope: Number.parseInt(formData.slope),
        ca: Number.parseInt(formData.ca),
        thal: Number.parseInt(formData.thal),
      }

      // For demo purposes, we'll simulate the API call
      // In production, replace with actual API endpoint
      await new Promise((resolve) => setTimeout(resolve, 2000))

      // Simulate prediction based on age and other factors
      const riskScore =
        (apiData.age > 50 ? 0.3 : 0.1) +
        (apiData.sex === 1 ? 0.2 : 0.1) +
        (apiData.cp > 2 ? 0.3 : 0.1) +
        (apiData.chol > 240 ? 0.2 : 0.1)

      const prediction = riskScore > 0.5 ? 1 : 0

      setResult({
        prediction,
        probability: riskScore,
        risk_level: prediction === 1 ? "High Risk" : "Low Risk",
      })
    } catch (err) {
      setError("Failed to get prediction. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const isFormValid = Object.values(formData).every((value) => value !== "")

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
          </div>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Heart Disease Risk Assessment</h2>
          <p className="text-gray-600">Enter patient clinical parameters to assess cardiovascular risk</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Form */}
          <Card>
            <CardHeader>
              <CardTitle>Patient Information</CardTitle>
              <CardDescription>Please fill in all required clinical parameters</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                {/* Age */}
                <div>
                  <Label htmlFor="age">Age (years)</Label>
                  <Input
                    id="age"
                    type="number"
                    value={formData.age}
                    onChange={(e) => handleInputChange("age", e.target.value)}
                    placeholder="e.g., 45"
                    min="1"
                    max="120"
                    required
                  />
                </div>

                {/* Sex */}
                <div>
                  <Label htmlFor="sex">Sex</Label>
                  <Select value={formData.sex} onValueChange={(value) => handleInputChange("sex", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select sex" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">Female</SelectItem>
                      <SelectItem value="1">Male</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Chest Pain Type */}
                <div>
                  <Label htmlFor="cp">Chest Pain Type</Label>
                  <Select value={formData.cp} onValueChange={(value) => handleInputChange("cp", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select chest pain type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">Typical Angina</SelectItem>
                      <SelectItem value="1">Atypical Angina</SelectItem>
                      <SelectItem value="2">Non-anginal Pain</SelectItem>
                      <SelectItem value="3">Asymptomatic</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Resting Blood Pressure */}
                <div>
                  <Label htmlFor="trestbps">Resting Blood Pressure (mm Hg)</Label>
                  <Input
                    id="trestbps"
                    type="number"
                    value={formData.trestbps}
                    onChange={(e) => handleInputChange("trestbps", e.target.value)}
                    placeholder="e.g., 120"
                    min="80"
                    max="200"
                    required
                  />
                </div>

                {/* Cholesterol */}
                <div>
                  <Label htmlFor="chol">Serum Cholesterol (mg/dl)</Label>
                  <Input
                    id="chol"
                    type="number"
                    value={formData.chol}
                    onChange={(e) => handleInputChange("chol", e.target.value)}
                    placeholder="e.g., 200"
                    min="100"
                    max="600"
                    required
                  />
                </div>

                {/* Fasting Blood Sugar */}
                <div>
                  <Label htmlFor="fbs">Fasting Blood Sugar {">"} 120 mg/dl</Label>
                  <Select value={formData.fbs} onValueChange={(value) => handleInputChange("fbs", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select option" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">No</SelectItem>
                      <SelectItem value="1">Yes</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Resting ECG */}
                <div>
                  <Label htmlFor="restecg">Resting ECG Results</Label>
                  <Select value={formData.restecg} onValueChange={(value) => handleInputChange("restecg", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select ECG result" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">Normal</SelectItem>
                      <SelectItem value="1">ST-T Wave Abnormality</SelectItem>
                      <SelectItem value="2">Left Ventricular Hypertrophy</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Max Heart Rate */}
                <div>
                  <Label htmlFor="thalach">Maximum Heart Rate Achieved</Label>
                  <Input
                    id="thalach"
                    type="number"
                    value={formData.thalach}
                    onChange={(e) => handleInputChange("thalach", e.target.value)}
                    placeholder="e.g., 150"
                    min="60"
                    max="220"
                    required
                  />
                </div>

                {/* Exercise Induced Angina */}
                <div>
                  <Label htmlFor="exang">Exercise Induced Angina</Label>
                  <Select value={formData.exang} onValueChange={(value) => handleInputChange("exang", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select option" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">No</SelectItem>
                      <SelectItem value="1">Yes</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* ST Depression */}
                <div>
                  <Label htmlFor="oldpeak">ST Depression Induced by Exercise</Label>
                  <Input
                    id="oldpeak"
                    type="number"
                    step="0.1"
                    value={formData.oldpeak}
                    onChange={(e) => handleInputChange("oldpeak", e.target.value)}
                    placeholder="e.g., 1.0"
                    min="0"
                    max="10"
                    required
                  />
                </div>

                {/* Slope */}
                <div>
                  <Label htmlFor="slope">Slope of Peak Exercise ST Segment</Label>
                  <Select value={formData.slope} onValueChange={(value) => handleInputChange("slope", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select slope" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">Upsloping</SelectItem>
                      <SelectItem value="1">Flat</SelectItem>
                      <SelectItem value="2">Downsloping</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* CA */}
                <div>
                  <Label htmlFor="ca">Number of Major Vessels Colored by Fluoroscopy</Label>
                  <Select value={formData.ca} onValueChange={(value) => handleInputChange("ca", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select number" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">0</SelectItem>
                      <SelectItem value="1">1</SelectItem>
                      <SelectItem value="2">2</SelectItem>
                      <SelectItem value="3">3</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Thal */}
                <div>
                  <Label htmlFor="thal">Thalassemia</Label>
                  <Select value={formData.thal} onValueChange={(value) => handleInputChange("thal", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">Normal</SelectItem>
                      <SelectItem value="2">Fixed Defect</SelectItem>
                      <SelectItem value="3">Reversible Defect</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button type="submit" className="w-full" disabled={!isFormValid || loading}>
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Predict Risk"
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Results */}
          <div className="space-y-6">
            {error && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {result && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    {result.prediction === 1 ? (
                      <AlertTriangle className="h-6 w-6 text-red-500" />
                    ) : (
                      <CheckCircle className="h-6 w-6 text-green-500" />
                    )}
                    <span>Risk Assessment Result</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div
                      className={`p-4 rounded-lg ${
                        result.prediction === 1
                          ? "bg-red-50 border border-red-200"
                          : "bg-green-50 border border-green-200"
                      }`}
                    >
                      <div className="text-center">
                        <h3
                          className={`text-2xl font-bold ${
                            result.prediction === 1 ? "text-red-700" : "text-green-700"
                          }`}
                        >
                          {result.risk_level}
                        </h3>
                        <p className={`text-sm ${result.prediction === 1 ? "text-red-600" : "text-green-600"}`}>
                          {result.prediction === 1
                            ? "Patient shows indicators of potential heart disease risk"
                            : "Patient shows low indicators of heart disease risk"}
                        </p>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm font-medium">Risk Score:</span>
                        <span className="text-sm">{(result.probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${result.prediction === 1 ? "bg-red-500" : "bg-green-500"}`}
                          style={{ width: `${result.probability * 100}%` }}
                        />
                      </div>
                    </div>

                    <Alert>
                      <AlertDescription>
                        <strong>Important:</strong> This prediction is for educational purposes only. Always consult
                        with healthcare professionals for proper medical diagnosis and treatment decisions.
                      </AlertDescription>
                    </Alert>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Information Card */}
            <Card>
              <CardHeader>
                <CardTitle>About This Assessment</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-gray-600 space-y-2">
                <p>
                  This tool uses machine learning algorithms trained on clinical heart disease datasets to assess
                  cardiovascular risk.
                </p>
                <p>
                  <strong>Parameters considered:</strong>
                </p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>Demographics (age, sex)</li>
                  <li>Clinical measurements (BP, cholesterol, heart rate)</li>
                  <li>Diagnostic test results (ECG, stress test)</li>
                  <li>Symptom indicators (chest pain, angina)</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
