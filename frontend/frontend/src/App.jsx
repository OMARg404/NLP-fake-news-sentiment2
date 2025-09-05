// App.jsx
import React from 'react';
import PredictForm from './components/PredictForm';
import './App.css';

export default function App() {
  return (
    <div className="App">
      {/* Navbar */}
      <nav className="navbar">
        <div className="logo"><a href="#">NewsCheck</a></div>
        <ul className="menu">
          <li><a href="#home">Home</a></li>
          <li><a href="#verify">Verify</a></li>
          <li><a href="#articles">Articles</a></li>
          <li><a href="#about">About</a></li>
        </ul>
      </nav>

      {/* News slider */}
      <div className="newsSlider">
        <span>
          Breaking: Global fact-check summit addresses rise of misinformation • 
          Researchers develop AI to detect fake news • NewsCheck beta launched worldwide
        </span>
      </div>

      {/* Home Section */}
      <section id="home" className="home section-glass">
        <div className="home-content">
          <h1>Stop Fake News in its Tracks</h1>
          <p className="home-subtitle">
            Empowering you with AI-powered tools to separate facts from misinformation.
          </p>

          <div className="features">
            <div className="feature-card">
              <h3>⚡ Fast</h3>
              <p>Instant results with just one click.</p>
            </div>
            <div className="feature-card">
              <h3>✅ Reliable</h3>
              <p>Built on trusted datasets and fact-check sources.</p>
            </div>
            <div className="feature-card">
              <h3>🤖 AI-Powered</h3>
              <p>Harnessing machine learning to catch misinformation.</p>
            </div>
          </div>

          <a href="#verify" className="cta-btn">Try It Now</a>
        </div>
      </section>

      {/* Verify Section */}
      <section id="verify" className="checker section-glass">
        <div className="checker-content">
          <h1>Verify News Credibility</h1>
          <p className="subtitle">
            In today's fast-paced information world, news can spread quickly — but not all of it is true. 
            Paste a news headline or paragraph below to see how our AI models evaluate its credibility, 
            helping you distinguish fact from fiction and make informed decisions.
          </p>

          <div id="result-box" className="result-box hidden" role="alert">
            <h2>Validation Result</h2>
            <p id="result-text"></p>
          </div>

          {/* Component for prediction */}
          <PredictForm />
        </div>
      </section>

      {/* Articles Section */}
      <section id="articles" className="articles section-glass">
        <h2>Latest Articles</h2>
        <p>
          Stay informed with our curated list of fact-checked news and articles. Learn how AI and research 
          can help you navigate the complex world of information and avoid misinformation traps.
        </p>
      </section>

      {/* About Section */}
      <section id="about" className="about section-glass">
        <h2>About NewsCheck</h2>
        <p>
          NewsCheck is dedicated to empowering individuals globally by providing tools and insights 
          to separate reliable journalism from misleading information, promoting a well-informed society.
        </p>
      </section>
    </div>
  );
}
