# Speech Tuner - Web Application

A professional voice and text analysis web application that detects emotions from voice recordings and analyzes sentiment from text input. Built with Flask and modern web technologies.

**Built with ❤️ by Tooba Jatoi**

## 🌟 Features

- 🎙️ **Voice Recording & Emotion Detection**: Record audio directly in the browser and analyze emotions
- 📝 **Text Sentiment Analysis**: Analyze sentiment from text input
- 📊 **Interactive Visualizations**: Beautiful charts and graphs for results
- 🎯 **Tone Recommendations**: Get personalized tone suggestions
- 💻 **Modern Responsive UI**: Works perfectly on desktop and mobile
- ☁️ **Cloud Ready**: Easy deployment to various platforms

## 🚀 Quick Start

### Local Development

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/speech-tuner.git
cd speech-tuner
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements_web.txt
```

4. **Run the application**:
```bash
python app.py
```

5. **Open your browser**:
   - Navigate to: http://localhost:5000
   - Start recording audio and analyzing text!

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Audio Processing**: Web Audio API, MediaRecorder
- **Charts**: Chart.js
- **Styling**: Custom CSS with modern design
- **AI/ML**: Transformers, scikit-learn, librosa

## 📦 Deployment Options

### 1. Heroku Deployment

1. **Create a Heroku account** and install Heroku CLI
2. **Set up your app**:
```bash
heroku create your-app-name
git add .
git commit -m "Initial deployment"
git push heroku main
```

3. **Set environment variables** (if needed):
```bash
heroku config:set SECRET_KEY=your-secret-key
```

### 2. Railway Deployment

1. **Connect your GitHub repository** to Railway
2. **Railway will automatically detect** the Flask app
3. **Deploy with one click** from the Railway dashboard

### 3. Render Deployment

1. **Create a Render account** and connect your GitHub repo
2. **Configure the service**:
   - Build Command: `pip install -r requirements_web.txt`
   - Start Command: `python app.py`
   - Environment: Python 3.10

### 4. Vercel Deployment

1. **Install Vercel CLI**:
```bash
npm i -g vercel
```

2. **Deploy**:
```bash
vercel
```

### 5. GitHub Pages (Static Version)

For a static version that can be deployed on GitHub Pages, see the `static/` directory.

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
SECRET_KEY=your-secret-key-here
FLASK_ENV=production
FLASK_DEBUG=False
```

### Customization

- **Colors**: Modify the CSS variables in `templates/index.html`
- **Models**: Update the model paths in the analyzer classes
- **Features**: Add new analysis features in `app.py`

## 📁 Project Structure

```
speech-tuner/
├── app.py                 # Main Flask application
├── requirements_web.txt   # Web app dependencies
├── templates/
│   └── index.html        # Main web interface
├── emotion_detector/     # Emotion detection module
├── sentiment_checker/    # Sentiment analysis module
├── speech_tuner/         # Speech tuning module
├── .github/workflows/    # GitHub Actions
├── Procfile             # Heroku deployment
├── runtime.txt          # Python version
└── README_WEB.md        # This file
```

## 🎨 Customization

### Adding New Features

1. **Backend**: Add new routes in `app.py`
2. **Frontend**: Update the HTML/JavaScript in `templates/index.html`
3. **Styling**: Modify the CSS in the `<style>` section

### Styling

The application uses a modern design system with:
- **Primary Color**: #2563EB (Blue)
- **Secondary Color**: #10B981 (Green)
- **Background**: Gradient from #667eea to #764ba2
- **Typography**: Inter font family

## 🔒 Security Considerations

- **HTTPS**: Always use HTTPS in production
- **CORS**: Configure CORS if needed for cross-origin requests
- **Rate Limiting**: Consider adding rate limiting for API endpoints
- **Input Validation**: Validate all user inputs

## 📊 Performance Optimization

- **Caching**: Implement caching for model predictions
- **CDN**: Use CDN for static assets
- **Compression**: Enable gzip compression
- **Image Optimization**: Optimize any images used

## 🐛 Troubleshooting

### Common Issues

1. **Audio Recording Not Working**:
   - Ensure HTTPS is enabled (required for microphone access)
   - Check browser permissions for microphone access

2. **Model Loading Errors**:
   - Verify all dependencies are installed
   - Check model file paths

3. **Deployment Issues**:
   - Ensure `Procfile` is in the root directory
   - Check Python version compatibility

### Debug Mode

Enable debug mode for development:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Gradio**: Original inspiration for the interface
- **Chart.js**: Beautiful chart visualizations
- **Font Awesome**: Icons
- **Inter Font**: Typography

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Contact the maintainers

---

**Happy Analyzing! 🎙️📊** 