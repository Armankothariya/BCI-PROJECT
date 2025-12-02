"""Enhanced Visual Styles for BCI App - Futuristic Neural Theme"""

def get_particle_background_js():
    """JavaScript for animated particle background"""
    return """
    <canvas id="particle-canvas" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1;"></canvas>
    <script>
    (function() {
        const canvas = document.getElementById('particle-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        class Particle {
            constructor() {
                this.reset();
                this.y = Math.random() * canvas.height;
            }
            reset() {
                this.x = Math.random() * canvas.width;
                this.y = 0;
                this.speed = Math.random() * 2 + 1;
                this.radius = Math.random() * 2 + 1;
                this.opacity = Math.random() * 0.5 + 0.3;
            }
            update() {
                this.y += this.speed;
                if (this.y > canvas.height) this.reset();
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(102, 126, 234, ${this.opacity})`;
                ctx.fill();
            }
        }
        
        const particles = Array.from({length: 100}, () => new Particle());
        
        function animate() {
            ctx.fillStyle = 'rgba(10, 14, 39, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => { p.update(); p.draw(); });
            requestAnimationFrame(animate);
        }
        animate();
        
        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    })();
    </script>
    """

def get_glassmorphism_css():
    """Enhanced CSS with glassmorphism and holographic effects"""
    return """
    <style>
    /* Dark neural background */
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0d1224 100%) !important; }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2), inset 0 0 30px rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    .glass-card:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4) !important;
        border-color: rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Holographic text */
    .holographic-text {
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite;
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
    }
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Emotion orb */
    .emotion-orb {
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.3) 20%, transparent 70%);
        box-shadow: 0 0 80px currentColor, inset 0 0 60px rgba(255,255,255,0.2);
        animation: hologramFloat 4s ease-in-out infinite;
        margin: 0 auto;
    }
    @keyframes hologramFloat {
        0%, 100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-20px) scale(1.05); }
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Neon glow */
    @keyframes neonPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.4); }
        50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.8), 0 0 60px rgba(118, 75, 162, 0.6); }
    }
    .neon-glow { animation: neonPulse 2s ease-in-out infinite; }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb) !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 12px; }
    ::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.05); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { box-shadow: 0 0 10px rgba(102, 126, 234, 0.5); }
    
    /* Metrics */
    .metric-card {
        background: rgba(102, 126, 234, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Animations */
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .page-transition { animation: slideIn 0.5s ease-out; }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
    }
    .floating { animation: float 3s ease-in-out infinite; }
    
    /* High confidence glow */
    @keyframes confidenceGlow {
        0%, 100% { box-shadow: 0 0 30px rgba(102, 126, 234, 0.6); }
        50% { box-shadow: 0 0 60px rgba(102, 126, 234, 1), 0 0 90px rgba(118, 75, 162, 0.8); }
    }
    .high-confidence { animation: confidenceGlow 2s ease-in-out infinite; }
    </style>
    """

def get_emotion_orb_html(emotion, confidence):
    """Holographic emotion orb"""
    colors = {"POSITIVE": "#667eea", "NEGATIVE": "#f5576c", "NEUTRAL": "#4facfe",
              "positive": "#667eea", "negative": "#f5576c", "neutral": "#4facfe"}
    color = colors.get(emotion, "#4facfe")
    size = 150 + (confidence * 100)
    glow_class = "high-confidence" if confidence > 0.8 else ""
    
    return f"""
    <div class="page-transition" style="text-align: center; padding: 2rem;">
        <div class="emotion-orb floating {glow_class}" style="width: {size}px; height: {size}px; color: {color};"></div>
        <h2 class="holographic-text">{emotion}</h2>
        <p style="font-size: 1.5rem; color: {color}; font-weight: 600;">{confidence*100:.1f}% Confidence</p>
    </div>
    """

def get_neural_network_svg():
    """Animated neural network visualization"""
    return """
    <svg width="100%" height="150" viewBox="0 0 600 150" style="margin: 1rem 0;">
        <defs>
            <linearGradient id="neuralGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                <stop offset="50%" style="stop-color:#764ba2;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#f093fb;stop-opacity:1" />
            </linearGradient>
            <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
        <!-- Input nodes -->
        <circle cx="80" cy="40" r="10" fill="url(#neuralGrad)" filter="url(#glow)"><animate attributeName="r" values="10;14;10" dur="2s" repeatCount="indefinite"/></circle>
        <circle cx="80" cy="75" r="10" fill="url(#neuralGrad)" filter="url(#glow)"><animate attributeName="r" values="10;14;10" dur="2s" begin="0.3s" repeatCount="indefinite"/></circle>
        <circle cx="80" cy="110" r="10" fill="url(#neuralGrad)" filter="url(#glow)"><animate attributeName="r" values="10;14;10" dur="2s" begin="0.6s" repeatCount="indefinite"/></circle>
        <!-- Hidden nodes -->
        <circle cx="300" cy="30" r="10" fill="url(#neuralGrad)" filter="url(#glow)"><animate attributeName="r" values="10;16;10" dur="2.5s" repeatCount="indefinite"/></circle>
        <circle cx="300" cy="65" r="10" fill="url(#neuralGrad)" filter="url(#glow)"><animate attributeName="r" values="10;16;10" dur="2.5s" begin="0.3s" repeatCount="indefinite"/></circle>
        <circle cx="300" cy="100" r="10" fill="url(#neuralGrad)" filter="url(#glow)"><animate attributeName="r" values="10;16;10" dur="2.5s" begin="0.6s" repeatCount="indefinite"/></circle>
        <!-- Output nodes -->
        <circle cx="520" cy="45" r="12" fill="#667eea" filter="url(#glow)"><animate attributeName="r" values="12;18;12" dur="3s" repeatCount="indefinite"/></circle>
        <circle cx="520" cy="85" r="12" fill="#f5576c" filter="url(#glow)"><animate attributeName="r" values="12;18;12" dur="3s" begin="0.5s" repeatCount="indefinite"/></circle>
        <circle cx="520" cy="125" r="12" fill="#4facfe" filter="url(#glow)"><animate attributeName="r" values="12;18;12" dur="3s" begin="1s" repeatCount="indefinite"/></circle>
        <!-- Connections -->
        <line x1="90" y1="40" x2="290" y2="30" stroke="url(#neuralGrad)" stroke-width="1.5" opacity="0.4"><animate attributeName="opacity" values="0.4;0.9;0.4" dur="2s" repeatCount="indefinite"/></line>
        <line x1="90" y1="75" x2="290" y2="65" stroke="url(#neuralGrad)" stroke-width="1.5" opacity="0.4"><animate attributeName="opacity" values="0.4;0.9;0.4" dur="2s" begin="0.5s" repeatCount="indefinite"/></line>
        <line x1="310" y1="30" x2="510" y2="45" stroke="url(#neuralGrad)" stroke-width="1.5" opacity="0.4"><animate attributeName="opacity" values="0.4;0.9;0.4" dur="2s" begin="1s" repeatCount="indefinite"/></line>
    </svg>
    """

def get_metric_card_html(title, value, subtitle="", color="#667eea"):
    """Glassmorphic metric card"""
    return f"""
    <div class="metric-card glass-card" style="border-color: {color};">
        <h4 style="color: {color}; margin: 0; font-size: 0.9rem; text-transform: uppercase;">{title}</h4>
        <div style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, {color}, {color}88); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0.5rem 0;">{value}</div>
        <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.85rem;">{subtitle}</p>
    </div>
    """

def get_section_header_html(title, subtitle=""):
    """Professional section header with underline"""
    return f"""
    <div class="page-transition" style="margin: 2rem 0 1rem 0;">
        <h2 style="font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; 
                   background: linear-gradient(135deg, #667eea, #764ba2); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {title}
        </h2>
        {f'<p style="color: rgba(255,255,255,0.7); margin: 0;">{subtitle}</p>' if subtitle else ''}
        <div style="height: 3px; width: 100px; background: linear-gradient(90deg, #667eea, #764ba2); 
                    border-radius: 5px; margin-top: 0.5rem;"></div>
    </div>
    """

def get_vertical_spacer(size="medium"):
    """Add consistent vertical spacing"""
    sizes = {"small": "1rem", "medium": "2rem", "large": "3rem"}
    return f'<div style="height: {sizes.get(size, "2rem")};"></div>'
