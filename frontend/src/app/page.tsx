"use client";

import { useState } from "react";
import axios from "axios";

interface DetectionResult {
  label: "hoax" | "fact";
  confidence: number;
  processed_text: string;
}

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDetect = async () => {
    if (!text.trim()) {
      setError("Please enter some text to analyze");
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post("http://localhost:8000/predict", {
        text: text
      });
      setResult(response.data);
    } catch (err) {
      setError("Error detecting hoax. Please try again.");
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Hoax Detection</h1>
          <p className="text-gray-600">Enter text to analyze whether it&apos;s a hoax or fact</p>
        </div>
        
        <div className="bg-white shadow-md rounded-lg p-6 mb-8">
          <div className="mb-6">
            <label htmlFor="text" className="block text-sm font-medium text-gray-700 mb-2">
              Text to analyze
            </label>
            <textarea
              id="text"
              rows={6}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Enter text here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>
          
          <div className="flex justify-center">
            <button
              onClick={handleDetect}
              disabled={loading}
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Analyzing..." : "Detect Hoax"}
            </button>
          </div>
        </div>
        
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-8">
            <div className="flex">
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}
        
        {result && (
          <div className="bg-white shadow-md rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Analysis Result</h2>
            
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-700 font-medium">Classification:</span>
                <span className={`font-semibold ${result.label === "hoax" ? "text-red-600" : "text-green-600"}`}>
                  {result.label.toUpperCase()}
                </span>
              </div>
              
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className={`h-2.5 rounded-full ${result.label === "hoax" ? "bg-red-600" : "bg-green-600"}`}
                  style={{width: `${result.confidence * 100}%`}}
                ></div>
              </div>
              
              <div className="text-right text-sm text-gray-500 mt-1">
                Confidence: {(result.confidence * 100).toFixed(2)}%
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Processed Text:</h3>
              <p className="text-gray-600 text-sm bg-gray-50 p-3 rounded border border-gray-200">
                {result.processed_text}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
