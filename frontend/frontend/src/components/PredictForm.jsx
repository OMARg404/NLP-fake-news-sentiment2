import React, { useState } from 'react';
import { predict } from '../api';
import ResultsCard from './ResultsCard';

export default function PredictForm() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text) return;
    setLoading(true);
    const res = await predict(text);
    setResult(res);
    setLoading(false);
  };

  return (
    <div className="predict-form">
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter news text..."
          rows={6}
          style={{ width: '100%' }}
        />
        <button
  type="submit"
  disabled={loading}
  style={{
    marginTop: '10px',
    padding: '12px 22px',
    border: 'none',
    borderRadius: '10px',
    background: 'linear-gradient(45deg, #ffd700, #ff8c00)',
    color: '#111',
    fontSize: '16px',
    fontWeight: '700',
    cursor: 'pointer',
  }}
>
  {loading ? 'Predicting...' : 'Predict'}
</button>
      </form>
      {result && <ResultsCard result={result} />}
    </div>
  );
}
