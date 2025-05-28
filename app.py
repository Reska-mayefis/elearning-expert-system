import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for deployment
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime
import json
import os

class ELearningExpertSystem:
    def __init__(self):
        self.setup_fuzzy_system()
        self.feedback_history = []
        
    def setup_fuzzy_system(self):
        """Setup sistem fuzzy dengan variabel input dan output"""
        
        # Definisi variabel input
        self.preferensi_visual = ctrl.Antecedent(np.arange(0, 101, 1), 'preferensi_visual')
        self.kebutuhan_penjelasan = ctrl.Antecedent(np.arange(0, 101, 1), 'kebutuhan_penjelasan')
        self.preferensi_interaktivitas = ctrl.Antecedent(np.arange(0, 101, 1), 'preferensi_interaktivitas')
        self.durasi_pembelajaran = ctrl.Antecedent(np.arange(0, 121, 1), 'durasi_pembelajaran')
        self.kompleksitas_materi = ctrl.Antecedent(np.arange(0, 101, 1), 'kompleksitas_materi')
        
        # Definisi variabel output
        self.rekomendasi = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')
        
        # Setup membership functions untuk preferensi visual
        self.preferensi_visual['rendah'] = fuzz.trapmf(self.preferensi_visual.universe, [0, 0, 20, 40])
        self.preferensi_visual['sedang'] = fuzz.trimf(self.preferensi_visual.universe, [20, 50, 80])
        self.preferensi_visual['tinggi'] = fuzz.trapmf(self.preferensi_visual.universe, [60, 80, 100, 100])
        
        # Setup membership functions untuk kebutuhan penjelasan
        self.kebutuhan_penjelasan['ringkas'] = fuzz.trapmf(self.kebutuhan_penjelasan.universe, [0, 0, 25, 45])
        self.kebutuhan_penjelasan['sedang'] = fuzz.trimf(self.kebutuhan_penjelasan.universe, [25, 50, 75])
        self.kebutuhan_penjelasan['mendalam'] = fuzz.trapmf(self.kebutuhan_penjelasan.universe, [55, 75, 100, 100])
        
        # Setup membership functions untuk preferensi interaktivitas
        self.preferensi_interaktivitas['rendah'] = fuzz.trapmf(self.preferensi_interaktivitas.universe, [0, 0, 25, 45])
        self.preferensi_interaktivitas['sedang'] = fuzz.trimf(self.preferensi_interaktivitas.universe, [25, 50, 75])
        self.preferensi_interaktivitas['tinggi'] = fuzz.trapmf(self.preferensi_interaktivitas.universe, [55, 75, 100, 100])
        
        # Setup membership functions untuk durasi pembelajaran (dalam menit)
        self.durasi_pembelajaran['singkat'] = fuzz.trapmf(self.durasi_pembelajaran.universe, [0, 0, 15, 25])
        self.durasi_pembelajaran['menengah'] = fuzz.trimf(self.durasi_pembelajaran.universe, [15, 30, 60])
        self.durasi_pembelajaran['panjang'] = fuzz.trapmf(self.durasi_pembelajaran.universe, [45, 60, 120, 120])
        
        # Setup membership functions untuk kompleksitas materi
        self.kompleksitas_materi['dasar'] = fuzz.trapmf(self.kompleksitas_materi.universe, [0, 0, 25, 45])
        self.kompleksitas_materi['menengah'] = fuzz.trimf(self.kompleksitas_materi.universe, [25, 50, 75])
        self.kompleksitas_materi['lanjut'] = fuzz.trapmf(self.kompleksitas_materi.universe, [55, 75, 100, 100])
        
        # Setup membership functions untuk output rekomendasi
        self.rekomendasi['video_pembelajaran'] = fuzz.trimf(self.rekomendasi.universe, [0, 10, 20])
        self.rekomendasi['infografis'] = fuzz.trimf(self.rekomendasi.universe, [15, 25, 35])
        self.rekomendasi['diagram_interaktif'] = fuzz.trimf(self.rekomendasi.universe, [30, 40, 50])
        self.rekomendasi['video_pendek'] = fuzz.trimf(self.rekomendasi.universe, [45, 55, 65])
        self.rekomendasi['slide_presentasi'] = fuzz.trimf(self.rekomendasi.universe, [60, 70, 80])
        self.rekomendasi['simulasi_visual'] = fuzz.trimf(self.rekomendasi.universe, [75, 85, 95])
        self.rekomendasi['animasi_konsep'] = fuzz.trimf(self.rekomendasi.universe, [85, 95, 100])
        
        self.setup_rules()
        
    def setup_rules(self):
        """Setup aturan fuzzy berdasarkan basis pengetahuan"""
        
        # Definisi aturan fuzzy
        self.rules = [
            # R1: Video Pembelajaran
            ctrl.Rule(
                self.preferensi_visual['tinggi'] & 
                self.kebutuhan_penjelasan['mendalam'] & 
                self.preferensi_interaktivitas['rendah'],
                self.rekomendasi['video_pembelajaran']
            ),
            
            # R2: Infografis
            ctrl.Rule(
                self.preferensi_visual['tinggi'] & 
                self.kebutuhan_penjelasan['ringkas'] & 
                self.preferensi_interaktivitas['rendah'],
                self.rekomendasi['infografis']
            ),
            
            # R3: Diagram Interaktif
            ctrl.Rule(
                self.preferensi_visual['tinggi'] & 
                self.preferensi_interaktivitas['tinggi'],
                self.rekomendasi['diagram_interaktif']
            ),
            
            # R4: Video Pendek
            ctrl.Rule(
                self.preferensi_visual['sedang'] & 
                self.durasi_pembelajaran['singkat'],
                self.rekomendasi['video_pendek']
            ),
            
            # R5: Slide Presentasi
            ctrl.Rule(
                self.preferensi_visual['sedang'] & 
                self.durasi_pembelajaran['menengah'],
                self.rekomendasi['slide_presentasi']
            ),
            
            # R6: Simulasi Visual
            ctrl.Rule(
                self.preferensi_visual['tinggi'] & 
                self.preferensi_interaktivitas['tinggi'] & 
                self.kompleksitas_materi['lanjut'],
                self.rekomendasi['simulasi_visual']
            ),
            
            # R9: Animasi Konsep
            ctrl.Rule(
                self.preferensi_visual['tinggi'] & 
                self.kompleksitas_materi['menengah'] & 
                self.kebutuhan_penjelasan['mendalam'],
                self.rekomendasi['animasi_konsep']
            ),
            
            # R12: Infografis Sederhana
            ctrl.Rule(
                self.preferensi_visual['tinggi'] & 
                self.kompleksitas_materi['dasar'] & 
                self.durasi_pembelajaran['singkat'],
                self.rekomendasi['infografis']
            )
        ]
        
        # Buat sistem kontrol
        self.recommendation_ctrl = ctrl.ControlSystem(self.rules)
        self.recommendation_sim = ctrl.ControlSystemSimulation(self.recommendation_ctrl)
    
    def get_recommendation(self, pref_visual, kebutuhan_penj, pref_interaktif, 
                          durasi_belajar, kompleks_materi):
        """Mendapatkan rekomendasi berdasarkan input pengguna"""
        
        try:
            # Set input values
            self.recommendation_sim.input['preferensi_visual'] = pref_visual
            self.recommendation_sim.input['kebutuhan_penjelasan'] = kebutuhan_penj
            self.recommendation_sim.input['preferensi_interaktivitas'] = pref_interaktif
            self.recommendation_sim.input['durasi_pembelajaran'] = durasi_belajar
            self.recommendation_sim.input['kompleksitas_materi'] = kompleks_materi
            
            # Compute result
            self.recommendation_sim.compute()
            
            # Get output value
            output_value = self.recommendation_sim.output['rekomendasi']
            
            # Map output value ke jenis rekomendasi
            recommendations = self.map_output_to_recommendation(output_value)
            
            # Hitung derajat keanggotaan untuk setiap input
            membership_details = self.calculate_membership_details(
                pref_visual, kebutuhan_penj, pref_interaktif, 
                durasi_belajar, kompleks_materi
            )
            
            return {
                'primary_recommendation': recommendations[0],
                'alternative_recommendations': recommendations[1:3],
                'confidence_score': round(output_value, 2),
                'membership_details': membership_details,
                'explanation': self.generate_explanation(
                    pref_visual, kebutuhan_penj, pref_interaktif,
                    durasi_belajar, kompleks_materi, recommendations[0]
                )
            }
            
        except Exception as e:
            return {
                'error': f"Error dalam perhitungan: {str(e)}",
                'primary_recommendation': 'slide_presentasi',
                'alternative_recommendations': ['video_pembelajaran', 'infografis'],
                'confidence_score': 0.5,
                'membership_details': {},
                'explanation': "Terjadi error dalam perhitungan, menggunakan rekomendasi default."
            }
    
    def map_output_to_recommendation(self, output_value):
        """Memetakan nilai output ke jenis rekomendasi"""
        
        recommendation_ranges = [
            (0, 20, 'video_pembelajaran'),
            (15, 35, 'infografis'),
            (30, 50, 'diagram_interaktif'),
            (45, 65, 'video_pendek'),
            (60, 80, 'slide_presentasi'),
            (75, 95, 'simulasi_visual'),
            (85, 100, 'animasi_konsep')
        ]
        
        # Hitung kedekatan dengan setiap range
        distances = []
        for min_val, max_val, rec_type in recommendation_ranges:
            center = (min_val + max_val) / 2
            distance = abs(output_value - center)
            distances.append((distance, rec_type))
        
        # Sort berdasarkan jarak terdekat
        distances.sort(key=lambda x: x[0])
        
        return [rec[1] for rec in distances[:5]]
    
    def calculate_membership_details(self, pref_visual, kebutuhan_penj, 
                                   pref_interaktif, durasi_belajar, kompleks_materi):
        """Menghitung detail derajat keanggotaan untuk setiap input"""
        
        details = {}
        
        # Preferensi Visual
        details['preferensi_visual'] = {
            'rendah': float(fuzz.interp_membership(self.preferensi_visual.universe, 
                                                 self.preferensi_visual['rendah'].mf, pref_visual)),
            'sedang': float(fuzz.interp_membership(self.preferensi_visual.universe, 
                                                 self.preferensi_visual['sedang'].mf, pref_visual)),
            'tinggi': float(fuzz.interp_membership(self.preferensi_visual.universe, 
                                                 self.preferensi_visual['tinggi'].mf, pref_visual))
        }
        
        # Kebutuhan Penjelasan
        details['kebutuhan_penjelasan'] = {
            'ringkas': float(fuzz.interp_membership(self.kebutuhan_penjelasan.universe, 
                                                  self.kebutuhan_penjelasan['ringkas'].mf, kebutuhan_penj)),
            'sedang': float(fuzz.interp_membership(self.kebutuhan_penjelasan.universe, 
                                                 self.kebutuhan_penjelasan['sedang'].mf, kebutuhan_penj)),
            'mendalam': float(fuzz.interp_membership(self.kebutuhan_penjelasan.universe, 
                                                   self.kebutuhan_penjelasan['mendalam'].mf, kebutuhan_penj))
        }
        
        # Preferensi Interaktivitas
        details['preferensi_interaktivitas'] = {
            'rendah': float(fuzz.interp_membership(self.preferensi_interaktivitas.universe, 
                                                 self.preferensi_interaktivitas['rendah'].mf, pref_interaktif)),
            'sedang': float(fuzz.interp_membership(self.preferensi_interaktivitas.universe, 
                                                 self.preferensi_interaktivitas['sedang'].mf, pref_interaktif)),
            'tinggi': float(fuzz.interp_membership(self.preferensi_interaktivitas.universe, 
                                                 self.preferensi_interaktivitas['tinggi'].mf, pref_interaktif))
        }
        
        return details
    
    def generate_explanation(self, pref_visual, kebutuhan_penj, pref_interaktif,
                           durasi_belajar, kompleks_materi, recommendation):
        """Generate penjelasan untuk rekomendasi"""
        
        explanations = {
            'video_pembelajaran': f"Berdasarkan preferensi visual yang tinggi ({pref_visual}%) dan kebutuhan penjelasan mendalam ({kebutuhan_penj}%), video pembelajaran akan memberikan visualisasi yang jelas dengan penjelasan komprehensif.",
            
            'infografis': f"Dengan preferensi visual tinggi ({pref_visual}%) dan kebutuhan informasi yang ringkas ({kebutuhan_penj}%), infografis akan menyajikan informasi secara visual dan mudah dipahami.",
            
            'diagram_interaktif': f"Preferensi visual tinggi ({pref_visual}%) dan interaktivitas tinggi ({pref_interaktif}%) membuat diagram interaktif menjadi pilihan ideal untuk eksplorasi konsep secara mandiri.",
            
            'video_pendek': f"Dengan preferensi visual sedang ({pref_visual}%) dan durasi pembelajaran singkat ({durasi_belajar} menit), video pendek memberikan informasi visual yang efisien.",
            
            'slide_presentasi': f"Preferensi visual sedang ({pref_visual}%) dengan durasi pembelajaran menengah ({durasi_belajar} menit) cocok dengan format slide presentasi yang terstruktur.",
            
            'simulasi_visual': f"Kombinasi preferensi visual tinggi ({pref_visual}%), interaktivitas tinggi ({pref_interaktif}%), dan materi kompleks ({kompleks_materi}%) membutuhkan simulasi visual untuk pemahaman mendalam.",
            
            'animasi_konsep': f"Preferensi visual tinggi ({pref_visual}%) dengan materi kompleks ({kompleks_materi}%) dan kebutuhan penjelasan mendalam ({kebutuhan_penj}%) ideal untuk animasi konsep yang detail."
        }
        
        return explanations.get(recommendation, "Rekomendasi berdasarkan analisis preferensi belajar visual Anda.")
    
    def add_feedback(self, user_input, recommendation, rating, comments=""):
        """Menambahkan feedback untuk pembelajaran sistem"""
        
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'recommendation': recommendation,
            'rating': rating,
            'comments': comments
        }
        
        self.feedback_history.append(feedback)
        
        # Simple adaptive learning - adjust rules based on feedback
        if rating >= 4:  # Good feedback
            print(f"Positive feedback received for {recommendation}")
        elif rating <= 2:  # Poor feedback
            print(f"Negative feedback received for {recommendation}")
    
    def get_feedback_analytics(self):
        """Analisis feedback untuk evaluasi sistem"""
        
        if not self.feedback_history:
            return {"message": "Belum ada feedback yang tersedia"}
        
        df = pd.DataFrame(self.feedback_history)
        
        analytics = {
            'total_feedback': len(self.feedback_history),
            'average_rating': df['rating'].mean(),
            'rating_distribution': df['rating'].value_counts().to_dict(),
            'recommendation_performance': df.groupby('recommendation')['rating'].agg(['mean', 'count']).to_dict(),
            'recent_feedback': self.feedback_history[-5:]  # 5 feedback terakhir
        }
        
        return analytics

# Flask Web Application
app = Flask(__name__)

# Initialize expert system
expert_system = ELearningExpertSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def get_recommendation():
    try:
        data = request.json
        
        pref_visual = float(data.get('preferensi_visual', 50))
        kebutuhan_penj = float(data.get('kebutuhan_penjelasan', 50))
        pref_interaktif = float(data.get('preferensi_interaktivitas', 50))
        durasi_belajar = float(data.get('durasi_pembelajaran', 30))
        kompleks_materi = float(data.get('kompleksitas_materi', 50))
        
        result = expert_system.get_recommendation(
            pref_visual, kebutuhan_penj, pref_interaktif,
            durasi_belajar, kompleks_materi
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        
        expert_system.add_feedback(
            data.get('user_input'),
            data.get('recommendation'),
            int(data.get('rating')),
            data.get('comments', '')
        )
        
        return jsonify({'message': 'Feedback berhasil disimpan'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics')
def get_analytics():
    analytics = expert_system.get_feedback_analytics()
    return jsonify(analytics)

@app.route('/health')
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({'status': 'healthy', 'message': 'E-Learning Expert System is running'})

# Fungsi untuk testing dan evaluasi
def test_system():
    """Fungsi untuk testing sistem dengan contoh kasus"""
    
    print("=== Testing Sistem Pakar E-Learning ===\n")
    
    # Test case 1: Rere (dari dokumen)
    print("Test Case 1 - Mahasiswa Rere:")
    print("Preferensi Visual: 85%, Kebutuhan Penjelasan: 75%")
    print("Preferensi Interaktivitas: 30%, Durasi: 25 menit")
    print("Kompleksitas Materi: 50% (menengah)")
    
    result1 = expert_system.get_recommendation(85, 75, 30, 25, 50)
    print(f"Rekomendasi: {result1['primary_recommendation']}")
    print(f"Confidence Score: {result1['confidence_score']}")
    print(f"Penjelasan: {result1['explanation']}\n")
    
    # Test case 2: Mahasiswa dengan preferensi berbeda
    print("Test Case 2 - Mahasiswa dengan preferensi interaktif tinggi:")
    print("Preferensi Visual: 90%, Kebutuhan Penjelasan: 40%")
    print("Preferensi Interaktivitas: 85%, Durasi: 45 menit")
    print("Kompleksitas Materi: 80% (lanjut)")
    
    result2 = expert_system.get_recommendation(90, 40, 85, 45, 80)
    print(f"Rekomendasi: {result2['primary_recommendation']}")
    print(f"Confidence Score: {result2['confidence_score']}")
    print(f"Penjelasan: {result2['explanation']}\n")
    
    # Test case 3: Mahasiswa dengan preferensi visual sedang
    print("Test Case 3 - Mahasiswa dengan preferensi visual sedang:")
    print("Preferensi Visual: 50%, Kebutuhan Penjelasan: 60%")
    print("Preferensi Interaktivitas: 40%, Durasi: 20 menit")
    print("Kompleksitas Materi: 30% (dasar)")
    
    result3 = expert_system.get_recommendation(50, 60, 40, 20, 30)
    print(f"Rekomendasi: {result3['primary_recommendation']}")
    print(f"Confidence Score: {result3['confidence_score']}")
    print(f"Penjelasan: {result3['explanation']}\n")

if __name__ == '__main__':
    # Set environment variables
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    # Run tests if in development mode
    if debug_mode:
        test_system()
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, debug=debug_mode)