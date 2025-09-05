import React from 'react';

export default function ResultsCard({ result }) {
  if (!result || !result.results) return null;

  return (
    <div
      className="results-card"
      style={{
        marginTop: '20px',
        padding: '20px',
        borderLeft: '6px solid #ffd700',
        background: 'rgba(255, 215, 0, 0.1)',
        color: '#111',
        fontFamily: 'Georgia, serif',
        fontSize: '17px',
        lineHeight: '1.6',
        borderRadius: '8px',
        textAlign: 'left',
      }}
    >
      <h3>Predictions by Each Model:</h3>
      <ul>
        {result.results.map((r, idx) => (
          <li key={idx}>
            <strong>{r.model}:</strong> {r.prediction}
          </li>
        ))}
      </ul>

      <h3 style={{ marginTop: '15px', color: '#0056b3' }}>
        Final Sentiment: {result.final_sentiment}
      </h3>
    </div>
  );
}
