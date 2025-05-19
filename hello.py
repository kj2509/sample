#!/usr/bin/env python
# Srirampur Mine Blast 2020 Sentiment Analyzer with HPLT Register Integration
# Analyzes sentiment around Srirampur RK 5B underground mine blast with register analysis
# September 2, 2020 as the temporal separator
# Pre-blast (before Sept 2, 2020) vs Post-blast (after Sept 2, 2020)

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
import time
import sys
import threading
from datetime import timezone

# Auto-install required libraries
required_packages = ['zstandard', 'tqdm', 'textblob', 'vaderSentiment']

for package in required_packages:
    try:
        if package == 'zstandard':
            import zstandard as zstd
            print(f"[OK] {package} available")
        elif package == 'tqdm':
            from tqdm.auto import tqdm
            print(f"[OK] {package} available")
        elif package == 'textblob':
            from textblob import TextBlob
            print(f"[OK] {package} available")
        elif package == 'vaderSentiment':
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            print(f"[OK] {package} available")
    except ImportError:
        print(f"[INSTALL] Installing {package}...")
        import subprocess
        if package == 'zstandard':
            subprocess.check_call(['pip', 'install', 'zstandard'])
            import zstandard as zstd
        elif package == 'tqdm':
            subprocess.check_call(['pip', 'install', 'tqdm'])
            from tqdm.auto import tqdm
        elif package == 'textblob':
            subprocess.check_call(['pip', 'install', 'textblob'])
            from textblob import TextBlob
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                nltk.download('brown', quiet=True)
            except:
                pass
        elif package == 'vaderSentiment':
            subprocess.check_call(['pip', 'install', 'vaderSentiment'])
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Font setup
def setup_indian_font():
    """Setup font for Indian languages with fallback"""
    import matplotlib.font_manager as fm
    
    indian_fonts = [
        'Noto Sans Devanagari',
        'Noto Sans Telugu',
        'Mangal',
        'Kokila', 
        'Aparajita',
        'Sanskrit Text',
        'Nirmala UI',
        'Arial Unicode MS',
        'DejaVu Sans'
    ]
    
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    
    font_found = None
    for font in indian_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            font_found = font
            print(f"[FONT] Using: {font}")
            break
    
    if not font_found:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("[FONT] Using system default")
    
    plt.rcParams['axes.unicode_minus'] = False
    return font_found

selected_font = setup_indian_font()

class SrirampurMineBlastAnalyzer:
    def __init__(self):
        # Srirampur mine blast date - September 2, 2020
        self.blast_date = datetime(2020, 9, 2, tzinfo=timezone.utc)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # HPLT Register Definitions - added to understand language varieties
        self.register_descriptions = {
        # Main categories
        "Lyrical": "Artistic/expressive content like poetry or lyrics",
        "Spoken": "Transcribed speech or oral content",
        "Interview": "Question-answer format interviews",
        "Interactive_discussion": "Discussions, comments, forums",
        "Narrative": "Storytelling content",
        "News_report": "Journalistic reporting",
        "Sports_report": "Sports-specific reporting",
        "Narrative_blog": "Blog-style narrative",
        "How-to": "Instructions, guides, tutorials",
        "Recipe": "Food recipes and cooking instructions",
        "Informational_persuasion": "Content designed to inform and persuade",
        "Description_with_intent_to_sell": "Marketing or sales content",
        "News_opinion_blog": "News with editorial content",
        
        # Sub-categories
        "Informational_description": "Basic informational content",
        "Encyclopedia_article": "Encyclopedia-style informational content",
        "Research_article": "Research-focused content",
        "Description_thing_person": "Content describing objects or people",
        "FAQ": "Frequently asked questions format",
        "Legal_terms": "Legal documents and terms",
        "Opinion": "Opinion pieces",
        "Review": "Product or service reviews",
        "Opinion_blog": "Blog-style opinion content",
        "Religious_blog": "Religious or denominational content",
        "Advice": "Advisory content",
        
        # Other
        "Machine-translated": "Machine-translated content",
        "unknown": "Content with unidentified register"
    }
    
    # Register abbreviations mapping
        self.register_abbreviations = {
        "LY": "Lyrical",
        "SP": "Spoken",
        "IT": "Interview",
        "ID": "Interactive_discussion",
        "NA": "Narrative",
        "NE": "News_report",
        "SR": "Sports_report",
        "NB": "Narrative_blog",
        "HI": "How-to",
        "RE": "Recipe",
        "IP": "Informational_persuasion",
        "DS": "Description_with_intent_to_sell", 
        "ED": "News_opinion_blog",
        
        # Sub-categories
        "IN": "Informational_description",
        "EN": "Encyclopedia_article",
        "RA": "Research_article",
        "DTP": "Description_thing_person", 
        "FI": "FAQ",
        "LT": "Legal_terms", 
        "OP": "Opinion",
        "RV": "Review",
        "OB": "Opinion_blog",
        "RS": "Religious_blog",
        "AV": "Advice",
        
        # Other
        "MT": "Machine-translated"
    }
    
    # Define main categories and subcategories
        self.main_register_categories = {
        "Lyrical", "Spoken", "Interview", "Interactive_discussion", 
        "Narrative", "News_report", "Sports_report", "Narrative_blog", 
        "How-to", "Recipe", "Informational_persuasion", 
        "Description_with_intent_to_sell", "News_opinion_blog"
    }
    
        self.sub_register_categories = {
        "Informational_description", "Encyclopedia_article", "Research_article",
        "Description_thing_person", "FAQ", "Legal_terms", "Opinion", "Review",
        "Opinion_blog", "Religious_blog", "Advice"
    }
        
        # Comprehensive mining and blast related keywords including rare earth metals
        self.mining_keywords = {
            'core_mining_hindi': [
                'खनन', 'कोयला', 'खदान', 'माइन', 'श्रीरामपुर', 'सिंगरेनी', 'मंचेरियल',
                'तेलंगाना', 'एसीसीएल', 'भूमिगत', 'विस्फोट', 'दुर्घटना', 'खनिज', 'अयस्क'
            ],
            'core_mining_english': [
                'mining', 'coal', 'mine', 'srirampur', 'singerani', 'mancherial',
                'telangana', 'sccl', 'underground', 'blast', 'explosion', 'accident',
                'mineral', 'ore', 'mining industry', 'mining operations'
            ],
            'mining_types_hindi': [
                'भूमिगत खनन', 'भूतल खनन', 'खुला खनन', 'पट्टी खनन', 'प्लेसर खनन',
                'पर्वत शिखर हटाना', 'ड्रिलिंग', 'विस्फोट', 'खदान', 'अयस्क निकासी',
                'भावी सर्वेक्षण', 'खदान विकास', 'शाफ्ट डुबाना', 'सुरंग', 'अन्वेषण ड्रिलिंग',
                'कोर सैंपलिंग', 'भूवैज्ञानिक सर्वेक्षण'
            ],
            'mining_types_english': [
                'open-pit mining', 'underground mining', 'surface mining', 'strip mining',
                'placer mining', 'mountaintop removal', 'drilling', 'blasting', 'quarrying',
                'ore extraction', 'prospecting', 'mine development', 'shaft sinking',
                'tunneling', 'exploration drilling', 'core sampling', 'geological survey'
            ],
            'rare_earth_metals_hindi': [
                'दुर्लभ पृथ्वी तत्व', 'लिथियम', 'खनिज निक्षेप', 'यूरेनियम', 'दुर्लभ पृथ्वी निष्कर्षण',
                'महत्वपूर्ण खनिज भंडार', 'दुर्लभ पृथ्वी परिशोधन', 'खनिज प्रसंस्करण सुविधा',
                'रेडियोधर्मी तत्व', 'भारी धातु', 'गैर-नवीकरणीय शोषण'
            ],
            'rare_earth_metals_english': [
                'rare earth elements', 'lithium', 'mineral deposits', 'uranium',
                'REE extraction', 'rare earth element extraction',
                'mineral processing facility', 'rare earth refining', 'critical minerals reserves',
                'rare earths legislation', 'heavy metal pollution', 'non-renewable exploitation'
            ],
            'mining_problems_hindi': [
                'एसिड खदान जल निकासी', 'खदान बंद', 'खनन परमिट', 'पर्यावरणीय प्रभाव',
                'भूजल संदूषण', 'संसाधन निष्कर्षण', 'खदान ढहना', 'गैस विस्फोट',
                'खदान दुर्घटना', 'खदान आग', 'खदान बाढ़', 'संसाधन राष्ट्रवाद',
                'खनन प्रतिबंध', 'अवैध खनन', 'भूमि अधिग्रहण', 'वनों की कटाई',
                'जैव विविधता हानि', 'धूल प्रदूषण', 'काला फेफड़ा रोग', 'विषाक्त रिसाव',
                'सुरक्षा उल्लंघन', 'आपदा प्रतिक्रिया', 'संघर्ष खनिज', 'खनन रॉयल्टी'
            ],
            'mining_problems_english': [
                'acid mine drainage', 'mine closure', 'mining permit', 'environmental impact',
                'groundwater contamination', 'resource extraction', 'mine collapse', 'gas explosion',
                'mine accident', 'mine fire', 'mine flooding', 'resource nationalism',
                'mining ban', 'illegal mining', 'land expropriation', 'deforestation',
                'biodiversity loss', 'dust pollution', 'black lung disease', 'BLD',
                'toxic spill', 'safety violation', 'disaster response', 'conflict minerals',
                'mining royalties'
            ],
            'environmental_impact_hindi': [
                'पर्यावरणीय विनाश', 'पारिस्थितिकी तंत्र ध्वंस', 'मिट्टी क्षरण', 'जलवायु प्रभाव',
                'बलिदान क्षेत्र', 'जलवायु उपनिवेशवाद', 'पर्यावरण हत्या', 'ग्रीनवाशिंग',
                'भारी धातु प्रदूषण', 'दूषित पानी', 'भूमि हड़पना', 'सहमति के बिना खनन',
                'स्वदेशी भूमि विस्थापन', 'पारिस्थितिक पदचिह्न', 'हरित खनन',
                'पर्यावरण सुरक्षा', 'खनन अपशिष्ट प्रबंधन'
            ],
            'environmental_impact_english': [
                'environmental destruction', 'ecosystem collapse', 'soil degradation', 'climate impact',
                'sacrifice zones', 'climate colonialism', 'ecocide', 'greenwashing',
                'contaminated water', 'land grabbing', 'mining without consent',
                'indigenous land displacement', 'ecological footprint', 'green mining',
                'environmental safety in mining', 'mining waste management'
            ],
            'corporate_governance_hindi': [
                'कॉर्पोरेट सामाजिक जिम्मेदारी', 'खनन कंपनियां', 'खनन से राजस्व',
                'खनन विनियम', 'खनन कानून', 'संसाधन प्रबंधन', 'पर्यावरणीय नीतियां',
                'खनन अधिकार', 'भूमि सुधार', 'भूमि अधिकार', 'खनन लाइसेंस',
                'संसाधन शासन', 'राज्य संचालित खनन', 'राज्य खनिज निक्षेप रजिस्टर'
            ],
            'corporate_governance_english': [
                'corporate social responsibility', 'mining companies', 'revenue from mining',
                'mining regulations', 'mining laws', 'resource management', 'environmental policies',
                'mining rights', 'land reclamation', 'land rights', 'mining licenses',
                'resource governance', 'state-run mining', 'state register of mineral deposits'
            ],
            'technical_terms_hindi': [
                'खनन इंजीनियरिंग', 'जीवाश्म संसाधन', 'खनिज प्रसंस्करण',
                'अयस्क परिवहन', 'पर्यावरणीय प्रभाव वक्तव्य', 'खनन पट्टा',
                'महत्वपूर्ण खनिज', 'प्रदूषण', 'संदूषण'
            ],
            'technical_terms_english': [
                'mining engineering', 'mineral resources', 'fossil resources', 'mineral processing',
                'ore transportation', 'environmental impact statement', 'EIS', 'mining lease',
                'critical minerals reserves', 'pollution', 'contamination'
            ],
            'blast_terms_hindi': [
                'धमाका', 'विस्फोट', 'दुर्घटना', 'हादसा', 'घटना', 'त्रासदी',
                'मृत्यु', 'मौत', 'जख्मी', 'घायल', 'पीड़ित', 'बचाव', 'राहत',
                'सुरक्षा', 'खतरा', 'चेतावनी'
            ],
            'blast_terms_english': [
                'blast', 'explosion', 'accident', 'incident', 'tragedy', 'mishap',
                'death', 'casualties', 'injured', 'victims', 'rescue', 'relief',
                'safety', 'danger', 'warning'
            ]
        }
        
        # All keywords combined
        self.all_keywords = []
        for category in self.mining_keywords.values():
            self.all_keywords.extend([k.lower() for k in category])

    def classify_blast_era_by_crawl_date(self, doc):
        """
        Classify document as pre-blast or post-blast using CRAWL DATE
        Logic: If webpage was crawled before Sept 2, 2020, it's pre-blast content
        """
        # Try to find crawl date in various possible fields
        crawl_date_fields = [
            'crawl_date', 'date_crawled', 'crawl_timestamp', 'fetch_date',
            'collection_date', 'harvest_date', 'ts'
        ]
        
        crawl_timestamp = None
        
        # First, try to find explicit crawl date fields
        for field in crawl_date_fields:
            if field in doc and doc[field]:
                crawl_timestamp = doc[field]
                break
        
        # If no explicit crawl date, check if 'ts' could be crawl time
        if not crawl_timestamp and 'ts' in doc:
            crawl_timestamp = doc['ts']
        
        if not crawl_timestamp:
            return 'unknown', None
        
        try:
            # Parse the crawl timestamp
            parsed_date = None
            
            # Try different parsing methods
            if isinstance(crawl_timestamp, str):
                # Method 1: Standard ISO format
                try:
                    parsed_date = pd.to_datetime(crawl_timestamp, errors='coerce')
                except:
                    pass
                
                # Method 2: Unix timestamp as string
                if pd.isna(parsed_date) and crawl_timestamp.replace('.', '').isdigit():
                    try:
                        # Try seconds first
                        unix_time = float(crawl_timestamp)
                        if unix_time > 1000000000:  # Reasonable timestamp (after year 2001)
                            if len(crawl_timestamp.replace('.', '')) > 10:  # Milliseconds
                                parsed_date = pd.to_datetime(int(unix_time * 1000), unit='ms')
                            else:  # Seconds
                                parsed_date = pd.to_datetime(int(unix_time), unit='s')
                    except:
                        pass
            
            elif isinstance(crawl_timestamp, (int, float)):
                # Direct numeric timestamp
                try:
                    if crawl_timestamp > 1000000000:
                        if crawl_timestamp > 10000000000:  # Milliseconds
                            parsed_date = pd.to_datetime(int(crawl_timestamp), unit='ms')
                        else:  # Seconds
                            parsed_date = pd.to_datetime(int(crawl_timestamp), unit='s')
                except:
                    pass
            
            if pd.isna(parsed_date) or parsed_date is None:
                return 'unknown', None
            
            # Classify based on blast date (September 2, 2020)
            if parsed_date < self.blast_date:
                return 'pre_blast', parsed_date
            else:
                return 'post_blast', parsed_date
                
        except Exception as e:
            return 'unknown', None

def get_register_probabilities(self, doc):
    """Extract HPLT register probabilities with better handling of main/sub categories"""
    register_probs = {}
    
    # Check for register fields in document
    register_fields = [
        'register', 'register_probs', 'predicted_register',
        'register_probabilities', 'text_type', 'register_labels'
    ]
    
    # Try direct register fields first
    for field in register_fields:
        if field in doc and doc[field]:
            # Convert various formats to our standard format
            if isinstance(doc[field], str):
                # Direct string value
                reg_value = doc[field].lower()
                # Check if it's an abbreviation
                if reg_value.upper() in self.register_abbreviations:
                    reg_name = self.register_abbreviations[reg_value.upper()]
                    register_probs = {reg_name: 1.0}
                else:
                    # Try to match to a register name
                    for reg_name in self.register_descriptions.keys():
                        if reg_value in reg_name.lower() or reg_name.lower() in reg_value:
                            register_probs = {reg_name: 1.0}
                            break
                    # If no match found, use as-is
                    if not register_probs:
                        register_probs = {reg_value: 1.0}
                break
                
            # Dictionary format
            elif isinstance(doc[field], dict):
                # Convert keys if needed
                converted_dict = {}
                for k, v in doc[field].items():
                    k_upper = k.upper() if len(k) <= 3 else k.title()
                    if k_upper in self.register_abbreviations:
                        converted_dict[self.register_abbreviations[k_upper]] = float(v)
                    else:
                        # Try fuzzy matching
                        matched = False
                        for reg_name in self.register_descriptions.keys():
                            if k.lower() in reg_name.lower() or reg_name.lower() in k.lower():
                                converted_dict[reg_name] = float(v)
                                matched = True
                                break
                        if not matched:
                            converted_dict[k] = float(v)
                
                register_probs = converted_dict
                break
                
            # List format (try to handle various formats)
            elif isinstance(doc[field], list):
                if all(isinstance(x, str) for x in doc[field]):
                    # Just category names
                    probs = 1.0 / len(doc[field])
                    register_probs = {}
                    for reg in doc[field]:
                        reg_upper = reg.upper() if len(reg) <= 3 else reg
                        if reg_upper in self.register_abbreviations:
                            register_probs[self.register_abbreviations[reg_upper]] = probs
                        else:
                            register_probs[reg] = probs
                    
                elif len(doc[field]) == 2 and isinstance(doc[field][0], str):
                    # [category, confidence] format
                    reg = doc[field][0]
                    reg_upper = reg.upper() if len(reg) <= 3 else reg
                    if reg_upper in self.register_abbreviations:
                        register_probs = {self.register_abbreviations[reg_upper]: float(doc[field][1])}
                    else:
                        register_probs = {reg: float(doc[field][1])}
                
                else:
                    # More complex list structure
                    for item in doc[field]:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            reg, prob = item
                            reg_upper = reg.upper() if len(reg) <= 3 else reg
                            if reg_upper in self.register_abbreviations:
                                register_probs[self.register_abbreviations[reg_upper]] = float(prob)
                            else:
                                register_probs[reg] = float(prob)
                break
    
    # If no register information found, attempt content-based classification
    if not register_probs and 'text' in doc and len(doc['text']) > 200:
        text = doc['text'].lower()
        register_probs = self._classify_register_from_content(text)
    
    # Return the found register probabilities or default
    return register_probs if register_probs else {'unknown': 1.0}     

    
    # Return the found register probabilities or default

class LiveStatusTracker:
    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            'files_downloaded': 0, 'total_files': 0, 'current_file': '',
            'docs_processed': 0, 'mining_docs_found': 0, 'current_keywords': [],
            'file_sizes': [], 'pre_blast_docs': 0, 'post_blast_docs': 0,
            'docs_with_dates': 0, 'docs_no_dates': 0, 'docs_with_registers': 0,
            'register_stats': defaultdict(int)
        }
        self.is_running = True
        
        # Hindi/Telugu transliteration (expanded for mining terms)
        self.transliteration = {
            # Core mining terms
            'खनन': 'khanan', 'कोयला': 'koyla', 'खदान': 'khadan', 'माइन': 'mine',
            'श्रीरामपुर': 'srirampur', 'सिंगरेनी': 'singareni', 'मंचेरियल': 'mancherial',
            'तेलंगाना': 'telangana', 'विस्फोट': 'visphot', 'दुर्घटना': 'durghatna',
            'धमाका': 'dhamaka', 'सुरक्षा': 'suraksha', 'श्रमिक': 'shramik',
            'सरकार': 'sarkar', 'मुआवजा': 'muaawaza', 'पर्यावरण': 'paryaavaran',
            'खनिज': 'khanij', 'अयस्क': 'ayask', 'भूमिगत': 'bhoomigat',
            
            # Mining types
            'भूमिगत खनन': 'bhoomigat khanan', 'भूतल खनन': 'bhootal khanan',
            'खुला खनन': 'khula khanan', 'पट्टी खनन': 'patti khanan',
            'ड्रिलिंग': 'drilling', 'सुरंग': 'surang',
            
            # Rare earth metals
            'दुर्लभ पृथ्वी तत्व': 'durlabh prithvi tatva', 'लिथियम': 'lithium',
            'यूरेनियम': 'uranium', 'भारी धातु': 'bhaari dhaatu',
            
            # Environmental terms
            'पर्यावरणीय विनाश': 'paryaavaraniya vinaash',
            'पारिस्थितिकी तंत्र': 'paaristhitiki tantra',
            'मिट्टी क्षरण': 'mitti ksharan', 'जलवायु प्रभाव': 'jalvaayu prabhaav',
            'दूषित पानी': 'dooshit paani', 'भूमि हड़पना': 'bhoomi hadapna',
            
            # Corporate terms
            'कॉर्पोरेट सामाजिक जिम्मेदारी': 'corporate saamaajik jimmedaari',
            'खनन कंपनियां': 'khanan kampaniyan', 'खनन अधिकार': 'khanan adhikaar',
            'खनन लाइसेंस': 'khanan license', 'संसाधन प्रबंधन': 'sansaadhan prabandhan',
            
            # Safety and problems
            'एसिड खदान': 'acid khadan', 'भूजल संदूषण': 'bhujal sandooshan',
            'खदान ढहना': 'khadan dhahna', 'गैस विस्फोट': 'gas visphot',
            'विषाक्त रिसाव': 'vishakt risaav', 'काला फेफड़ा रोग': 'kaala phephda rog',
            'धूल प्रदूषण': 'dhool pradooshan', 'सुरक्षा उल्लंघन': 'suraksha ullanghan'
        }
    
    def transliterate_hindi(self, text):
        """Simple transliteration for display"""
        if selected_font and 'DejaVu Sans' not in selected_font:
            return text
        
        result = text
        for hindi, roman in self.transliteration.items():
            result = result.replace(hindi, f"{roman}({hindi})")
        return result
    
    def start_live_updates(self):
        def update_loop():
            while self.is_running:
                self.print_live_status()
                time.sleep(2)
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def stop_live_updates(self):
        self.is_running = False
        if hasattr(self, 'update_thread'):
            self.update_thread.join(timeout=1)
    
    def print_live_status(self):
        elapsed = time.time() - self.start_time
        print(f"\r{' ' * 120}", end='')
        
        parts = [f"Time: {elapsed:.1f}s"]
        
        if self.stats['current_file']:
            file_name = os.path.basename(self.stats['current_file'])[:15]
            parts.append(f"File: {file_name}")
        
        if self.stats['docs_processed'] > 0:
            rate = self.stats['docs_processed'] / elapsed
            parts.append(f"Docs: {self.stats['docs_processed']:,} ({rate:.0f}/s)")
        
        if self.stats['mining_docs_found'] > 0:
            percentage = (self.stats['mining_docs_found'] / max(1, self.stats['docs_processed'])) * 100
            parts.append(f"Mining: {self.stats['mining_docs_found']:,} ({percentage:.1f}%)")
            
            if self.stats['pre_blast_docs'] > 0 or self.stats['post_blast_docs'] > 0:
                parts.append(f"Pre: {self.stats['pre_blast_docs']} | Post: {self.stats['post_blast_docs']}")
                
            if self.stats['docs_with_dates'] > 0:
                date_pct = (self.stats['docs_with_dates'] / self.stats['mining_docs_found']) * 100
                parts.append(f"Dates: {date_pct:.0f}%")
                
            if self.stats['docs_with_registers'] > 0:
                reg_pct = (self.stats['docs_with_registers'] / self.stats['mining_docs_found']) * 100
                parts.append(f"Reg: {reg_pct:.0f}%")
        
        print(f"\r{' | '.join(parts)}", end='', flush=True)
    
    def update_file_progress(self, file_name, files_done, total_files):
        self.stats['current_file'] = file_name
        self.stats['files_downloaded'] = files_done
        self.stats['total_files'] = total_files
    
    def update_processing_progress(self, docs_processed, mining_found, latest_keywords=None):
        self.stats['docs_processed'] = docs_processed
        self.stats['mining_docs_found'] = mining_found
        if latest_keywords:
            self.stats['current_keywords'].extend(latest_keywords)
            self.stats['current_keywords'] = self.stats['current_keywords'][-10:]
    
    def print_final_summary(self):
        self.stop_live_updates()
        elapsed = time.time() - self.start_time
        print(f"\n\n{'='*60}")
        print("[SUMMARY] Final Analysis Results")
        print(f"{'='*60}")
        print(f"[TIME] Total processing: {elapsed:.1f} seconds")
        print(f"[FILES] Processed: {self.stats['files_downloaded']}/{self.stats['total_files']}")
        print(f"[DOCS] Total processed: {self.stats['docs_processed']:,}")
        print(f"[FOUND] Mining documents: {self.stats['mining_docs_found']:,}")
        print(f"[DATES] With crawl dates: {self.stats['docs_with_dates']:,} ({self.stats['docs_with_dates']/max(1,self.stats['mining_docs_found'])*100:.1f}%)")
        print(f"[REGISTERS] With register labels: {self.stats['docs_with_registers']:,} ({self.stats['docs_with_registers']/max(1,self.stats['mining_docs_found'])*100:.1f}%)")
        print(f"[PRE-BLAST] Documents (crawled before Sept 2, 2020): {self.stats['pre_blast_docs']:,}")
        print(f"[POST-BLAST] Documents (crawled after Sept 2, 2020): {self.stats['post_blast_docs']:,}")
        
        # Print register distribution
        if self.stats['docs_with_registers'] > 0:
            print(f"\n[REGISTER DISTRIBUTION]")
            for register, count in sorted(self.stats['register_stats'].items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / self.stats['docs_with_registers']) * 100
                    print(f"  - {register}: {count} docs ({percentage:.1f}%)")
        
        if self.stats['docs_processed'] > 0:
            success_rate = (self.stats['mining_docs_found'] / self.stats['docs_processed']) * 100
            processing_rate = self.stats['docs_processed'] / elapsed
            print(f"\n[RATE] Success: {success_rate:.2f}%")
            print(f"[SPEED] Processing: {processing_rate:.0f} docs/second")
        
        if self.stats['file_sizes']:
            avg_size = sum(self.stats['file_sizes']) / len(self.stats['file_sizes'])
            print(f"[SIZE] Average file size: {avg_size:.1f} MB")
        print(f"{'='*60}\n")

class SrirampurMineBlastSentimentAnalyzer:
    def __init__(self, max_files=8, max_docs_per_file=25000, download_chunks_mb=15, use_registers=True):
        self.max_files = max_files
        self.max_docs_per_file = max_docs_per_file
        self.download_chunks_mb = download_chunks_mb
        self.use_registers = use_registers  # Flag to control register analysis
        self.status_tracker = LiveStatusTracker()
        self.blast_analyzer = SrirampurMineBlastAnalyzer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    def is_mining_related(self, text):
        """Check if text is related to mining, including rare earth metals and comprehensive mining terms"""
        text_lower = text.lower()
        
        # Check for core mining keywords
        core_mining_found = False
        keywords_found = []
        
        # Check for specific entities/companies/locations
        srirampur_found = False
        company_found = False
        location_found = False
        
        # Check for mining operations
        mining_operations_found = False
        rare_earth_found = False
        environmental_found = False
        
        # Check all Hindi keyword categories
        for category in ['core_mining_hindi', 'mining_types_hindi', 'rare_earth_metals_hindi', 
                        'mining_problems_hindi', 'environmental_impact_hindi', 'corporate_governance_hindi',
                        'technical_terms_hindi', 'blast_terms_hindi']:
            for keyword in self.blast_analyzer.mining_keywords[category]:
                if keyword in text:
                    keywords_found.append(keyword)
                    
                    # Categorize findings
                    if category == 'core_mining_hindi':
                        if keyword in ['श्रीरामपुर', 'सिंगरेनी']:
                            srirampur_found = True
                        elif keyword in ['खनन', 'कोयला', 'खदान', 'माइन']:
                            core_mining_found = True
                    elif category == 'mining_types_hindi':
                        mining_operations_found = True
                    elif category == 'rare_earth_metals_hindi':
                        rare_earth_found = True
                    elif category in ['mining_problems_hindi', 'environmental_impact_hindi']:
                        environmental_found = True
        
        # Check all English keyword categories
        for category in ['core_mining_english', 'mining_types_english', 'rare_earth_metals_english',
                        'mining_problems_english', 'environmental_impact_english', 'corporate_governance_english',
                        'technical_terms_english', 'blast_terms_english']:
            for keyword in self.blast_analyzer.mining_keywords[category]:
                if keyword.lower() in text_lower:
                    keywords_found.append(keyword)
                    
                    # Categorize findings
                    if category == 'core_mining_english':
                        if keyword.lower() in ['srirampur', 'singerani', 'sccl']:
                            srirampur_found = True
                        elif keyword.lower() in ['mining', 'coal', 'mine']:
                            core_mining_found = True
                        elif keyword.lower() in ['mancherial', 'telangana']:
                            location_found = True
                    elif category == 'mining_types_english':
                        mining_operations_found = True
                    elif category == 'rare_earth_metals_english':
                        rare_earth_found = True
                    elif category in ['mining_problems_english', 'environmental_impact_english']:
                        environmental_found = True
                    elif category == 'corporate_governance_english':
                        company_found = True
        
        # Document is relevant if:
        # 1. Mentions Srirampur mine specifically, OR
        # 2. Has core mining terms + any other category, OR  
        # 3. Has rare earth metals terms, OR
        # 4. Has multiple mining-related keywords from different categories
        
        relevance_score = 0
        if srirampur_found: relevance_score += 3
        if core_mining_found: relevance_score += 2
        if mining_operations_found: relevance_score += 1
        if rare_earth_found: relevance_score += 2
        if environmental_found: relevance_score += 1
        if company_found: relevance_score += 1
        if location_found: relevance_score += 1
        
        # Consider document relevant if relevance score >= 2 or has sufficient keywords
        is_relevant = (relevance_score >= 2 or 
                      len(keywords_found) >= 2 or 
                      srirampur_found or 
                      rare_earth_found)
        
        return is_relevant, keywords_found[:10]  # Limit to top 10 keywords

    def analyze_sentiment(self, text):
        """Analyze sentiment of text using multiple methods"""
        # VADER Sentiment (good for social media, informal text)
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment (good for formal text)
        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
        except:
            textblob_polarity = 0
            textblob_subjectivity = 0
        
        # Combined sentiment classification with lower thresholds
        compound_score = vader_scores['compound']
        
        # More sensitive thresholds for better classification
        if compound_score >= 0.02:  # Lowered from 0.05
            sentiment = 'positive'
        elif compound_score <= -0.02:  # Lowered from -0.05
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'vader_compound': compound_score,
            'vader_pos': vader_scores['pos'],
            'vader_neu': vader_scores['neu'],
            'vader_neg': vader_scores['neg'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'confidence': abs(compound_score)
        }
    
    def classify_blast_era(self, doc):
        """Use crawl date for blast era classification"""
        era, crawl_date = self.blast_analyzer.classify_blast_era_by_crawl_date(doc)
        return era, crawl_date
    
    def download_in_chunks(self, url, filename, chunk_mb=15):
        chunk_size = chunk_mb * 1024 * 1024
        downloaded_size = 0
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if downloaded_size >= chunk_size:
                            break
            return downloaded_size
        except Exception as e:
            print(f"\n[ERROR] Download failed: {e}")
            return 0
    
 
    def stream_process_zst_file(self, filepath):
        mining_docs = []
        docs_processed = 0
        latest_keywords = []
        pre_blast_count = 0
        post_blast_count = 0
        docs_with_dates = 0
        docs_with_registers = 0
        register_counts = defaultdict(int)
        
        try:
            chunk_size = 1024 * 1024
            
            with open(filepath, 'rb') as fh:
                dctx = zstd.ZstdDecompressor()
                stream_reader = dctx.stream_reader(fh)
                buffer = ""
                
                while docs_processed < self.max_docs_per_file:
                    chunk = stream_reader.read(chunk_size)
                    if not chunk:
                        break
                    
                    try:
                        text_chunk = chunk.decode('utf-8')
                        buffer += text_chunk
                    except UnicodeDecodeError:
                        try:
                            text_chunk = chunk.decode('utf-8', errors='ignore')
                            buffer += text_chunk
                        except:
                            continue
                    
                    lines = buffer.split('\n')
                    buffer = lines[-1]
                    
                    for line in lines[:-1]:
                        if not line.strip():
                            continue
                        
                        try:
                            doc = json.loads(line.strip())
                            docs_processed += 1
                            
                            if 'text' in doc and len(doc['text']) > 100:
                                is_relevant, keywords_found = self.is_mining_related(doc['text'])
                                
                                if is_relevant:
                                    # Extract register information if available
                                    register_info = {}
                                    if self.use_registers:
                                        register_info = self.blast_analyzer.get_register_probabilities(doc)
                                        # Count register distribution
                                        if register_info and any(k != 'unknown' for k in register_info.keys()):
                                            docs_with_registers += 1
                                            # Get most likely register
                                            most_likely_register = max(register_info.items(), key=lambda x: x[1])[0]
                                            register_counts[most_likely_register] += 1
                                    
                                    # Use crawl date classification
                                    blast_era, crawl_date = self.classify_blast_era(doc)
                                    
                                    # Count documents with valid crawl dates
                                    if blast_era == 'pre_blast':
                                        pre_blast_count += 1
                                        docs_with_dates += 1
                                    elif blast_era == 'post_blast':
                                        post_blast_count += 1
                                        docs_with_dates += 1
                                    
                                    # Analyze sentiment
                                    sentiment_analysis = self.analyze_sentiment(doc['text'])
                                    
                                    mining_doc = {
                                        'text': doc['text'][:2000],  # First 2000 chars
                                        'url': doc.get('u', ''),
                                        'original_timestamp': doc.get('ts', ''),  # Keep original for reference
                                        'crawl_date': crawl_date.isoformat() if crawl_date else None,
                                        'lang_prob': doc.get('prob', [1.0])[0] if doc.get('prob') else 1.0,
                                        'keywords_found': keywords_found,
                                        'blast_era': blast_era,
                                        'sentiment': sentiment_analysis['sentiment'],
                                        'sentiment_score': sentiment_analysis['vader_compound'],
                                        'confidence': sentiment_analysis['confidence'],
                                        'vader_scores': {
                                            'pos': sentiment_analysis['vader_pos'],
                                            'neu': sentiment_analysis['vader_neu'],
                                            'neg': sentiment_analysis['vader_neg']
                                        },
                                        'textblob_polarity': sentiment_analysis['textblob_polarity'],
                                        'register_info': register_info
                                    }
                                    mining_docs.append(mining_doc)
                                    latest_keywords.extend(keywords_found)
                            
                            if docs_processed % 500 == 0:  # Update every 500 docs
                                self.status_tracker.stats['pre_blast_docs'] = pre_blast_count
                                self.status_tracker.stats['post_blast_docs'] = post_blast_count
                                self.status_tracker.stats['docs_with_dates'] = docs_with_dates
                                self.status_tracker.stats['docs_with_registers'] = docs_with_registers
                                self.status_tracker.stats['register_stats'] = register_counts
                                self.status_tracker.update_processing_progress(
                                    docs_processed, len(mining_docs), latest_keywords[-5:] if latest_keywords else None
                                )
                                latest_keywords = []
                        except:
                            continue
                
                self.status_tracker.stats['pre_blast_docs'] = pre_blast_count
                self.status_tracker.stats['post_blast_docs'] = post_blast_count
                self.status_tracker.stats['docs_with_dates'] = docs_with_dates
                self.status_tracker.stats['docs_with_registers'] = docs_with_registers
                self.status_tracker.stats['register_stats'] = register_counts
                self.status_tracker.update_processing_progress(
                    docs_processed, len(mining_docs), latest_keywords[-5:] if latest_keywords else None
                )
                
        except Exception as e:
            print(f"\n[ERROR] Processing file: {e}")
        
        return mining_docs

    def download_and_analyze(self):
        print("[START] Srirampur Mine Blast 2020 Sentiment Analysis with HPLT Register Integration")
        print("="*60)
        print("[EVENT] Srirampur RK 5B Mine Blast (September 2, 2020)")
        print("[LOCATION] Mancherial district, Telangana, India")
        print("[COMPANY] Singerani Collieries Company Limited (SCCL)")
        print("[LOGIC] Documents crawled before Sept 2, 2020 = Pre-Blast")
        print("[LOGIC] Documents crawled after Sept 2, 2020 = Post-Blast")
        print("[FILTER] Excludes documents with unknown crawl dates")
        print("[FOCUS] Mining discourse, safety, rare earth metals, and community impact")
        print("[SCOPE] Comprehensive mining operations and environmental impact")
        if self.use_registers:
            print("[HPLT] Analyzing text registers from HPLT corpus")
            print("[REGISTERS] News, opinion, informational, instructional, promotional, etc.")
        print("="*60)
        
        self.status_tracker.start_live_updates()
        
        # Get HPLT Hindi file list (mining discussions likely in Hindi/Telugu)
        # For register-annotated files, we can either:
        # 1. Use the regular files and look for register fields in them
        # 2. Use the labels_ prefixed files if downloaded using the HPLT bash script
        map_url = "https://data.hplt-project.org/two/cleaned/hin_Deva_map.txt"
        
        try:
            response = requests.get(map_url, timeout=15)
            response.raise_for_status()
            files = response.text.strip().split('\n')
            files = [f for f in files if f.strip()][:self.max_files]
            
            self.status_tracker.stats['total_files'] = len(files)
            print(f"\n[FILES] Processing {len(files)} HPLT files")
            
        except Exception as e:
            print(f"[ERROR] Getting file list: {e}")
            self.status_tracker.stop_live_updates()
            return []
        
        all_mining_docs = []
        
        for i, file_url in enumerate(files):
            # Check if we're using register-labeled files
            if self.use_registers and "labels_" not in file_url and os.path.exists(f"labels_{os.path.basename(file_url)}"):
                # If we've already downloaded register-labeled files using the HPLT bash script
                print(f"\n[REGISTER] Using local register-labeled file for {os.path.basename(file_url)}")
                temp_file = f"labels_{os.path.basename(file_url)}"
                self.status_tracker.update_file_progress(temp_file, i, len(files))
            else:
                # Otherwise download directly from HPLT
                self.status_tracker.update_file_progress(file_url, i, len(files))
                temp_file = f"temp_mining_{i}.jsonl.zst"
                print(f"\n[DOWNLOAD] File {i+1}/{len(files)}...")
                
                downloaded_size = self.download_in_chunks(file_url, temp_file, self.download_chunks_mb)
                
                if downloaded_size == 0:
                    print(f"\n[FAILED] Could not download file {i+1}")
                    continue
                
                size_mb = downloaded_size / (1024 * 1024)
                self.status_tracker.stats['file_sizes'].append(size_mb)
                print(f"[OK] Downloaded {size_mb:.1f}MB")
            
            print(f"[PROCESS] Analyzing file {i+1}...")
            mining_docs = self.stream_process_zst_file(temp_file)
            
            if mining_docs:
                all_mining_docs.extend(mining_docs)
                print(f"\n[FOUND] {len(mining_docs)} mining documents in file {i+1}")
            else:
                print(f"\n[EMPTY] No mining documents in file {i+1}")
            
            # Only remove temp files we created (not labels_ files that might have been manually created)
            if temp_file.startswith("temp_") and os.path.exists(temp_file):
                os.remove(temp_file)
        
        self.status_tracker.update_file_progress("Complete", len(files), len(files))
        self.status_tracker.print_final_summary()
        
        return all_mining_docs
    
    def analyze_sentiment_trends(self, docs):
        print("[ANALYSIS] Computing mining discourse sentiment trends...")
        
        if not docs:
            print("[ERROR] No documents to analyze")
            return None, {}, {}
        
        df = pd.DataFrame(docs)
        
        # Filter out documents with unknown crawl dates
        print(f"[INFO] Total documents before filtering: {len(df)}")
        df = df[df['blast_era'] != 'unknown'].copy()
        print(f"[INFO] Documents with valid crawl dates: {len(df)}")
        
        if len(df) == 0:
            print("[ERROR] No documents with valid crawl dates found!")
            return None, {}, {}
        
        # Basic analysis
        print("[STEP] Analyzing document properties...")
        df['text_length'] = df['text'].str.len()
        df['has_hindi'] = df['text'].str.contains('[\u0900-\u097F]', regex=True)
        df['has_telugu'] = df['text'].str.contains('[\u0C00-\u0C7F]', regex=True)
        
        # Use crawl_date for temporal analysis
        df['crawl_datetime'] = pd.to_datetime(df['crawl_date'], errors='coerce')
        df['year'] = df['crawl_datetime'].dt.year
        df['month'] = df['crawl_datetime'].dt.month
        
        # Extract register information if available
        print("[STEP] Processing register information...")
        if self.use_registers and 'register_info' in df.columns:
            # Extract dominant register for each document
            df['dominant_register'] = df['register_info'].apply(
                lambda x: max(x.items(), key=lambda k: k[1])[0] if isinstance(x, dict) and x else 'unknown'
            )
            # Extract confidence for dominant register
            df['register_confidence'] = df['register_info'].apply(
                lambda x: max(x.items(), key=lambda k: k[1])[1] if isinstance(x, dict) and x else 0.0
            )
            # Count registers
            register_counts = df['dominant_register'].value_counts()
            print(f"[REGISTERS] Found register information in {len(df[df['dominant_register'] != 'unknown'])} documents")
            for reg, count in register_counts.items():
                print(f"  - {reg}: {count} documents")
        else:
            df['dominant_register'] = 'unknown'
            df['register_confidence'] = 0.0
            print("[REGISTERS] No register information available")
        
        # Keyword analysis
        print("[STEP] Extracting keyword patterns...")
        keyword_counts = defaultdict(int)
        for doc in docs:
            if 'keywords_found' in doc and doc['keywords_found'] and doc.get('blast_era') != 'unknown':
                for keyword in doc['keywords_found']:
                    keyword_counts[keyword] += 1
        
        keywords = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:25])
        
        # Sentiment analysis by era - based on CRAWL DATE (excluding unknowns)
        print("[STEP] Analyzing sentiment by blast era (only valid crawl dates)...")
        sentiment_analysis = {}
        
        # Overall sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        sentiment_analysis['overall'] = sentiment_counts.to_dict()
        
        # By blast era - only pre_blast and post_blast
        pre_blast = df[df['blast_era'] == 'pre_blast']
        post_blast = df[df['blast_era'] == 'post_blast']
        
        print(f"[INFO] Pre-Blast (crawled before Sept 2, 2020): {len(pre_blast)}")
        print(f"[INFO] Post-Blast (crawled after Sept 2, 2020): {len(post_blast)}")
        
        sentiment_analysis['by_era'] = {
            'pre_blast': {
                'total': len(pre_blast),
                'positive': len(pre_blast[pre_blast['sentiment'] == 'positive']),
                'neutral': len(pre_blast[pre_blast['sentiment'] == 'neutral']),
                'negative': len(pre_blast[pre_blast['sentiment'] == 'negative']),
                'avg_score': pre_blast['sentiment_score'].mean() if len(pre_blast) > 0 else 0
            },
            'post_blast': {
                'total': len(post_blast),
                'positive': len(post_blast[post_blast['sentiment'] == 'positive']),
                'neutral': len(post_blast[post_blast['sentiment'] == 'neutral']),
                'negative': len(post_blast[post_blast['sentiment'] == 'negative']),
                'avg_score': post_blast['sentiment_score'].mean() if len(post_blast) > 0 else 0
            }
        }
        
        # By register (if available)
        if self.use_registers and 'dominant_register' in df.columns:
            sentiment_analysis['by_register'] = {}
            for register in df['dominant_register'].unique():
                if register and register != 'unknown':
                    register_docs = df[df['dominant_register'] == register]
                    if len(register_docs) > 0:
                        sentiment_analysis['by_register'][register] = {
                            'total': len(register_docs),
                            'positive': len(register_docs[register_docs['sentiment'] == 'positive']),
                            'neutral': len(register_docs[register_docs['sentiment'] == 'neutral']),
                            'negative': len(register_docs[register_docs['sentiment'] == 'negative']),
                            'avg_score': register_docs['sentiment_score'].mean()
                        }
            
            # Compare pre/post blast sentiment by register
            sentiment_analysis['register_era_comparison'] = {}
            for register in df['dominant_register'].unique():
                if register and register != 'unknown':
                    pre_reg = pre_blast[pre_blast['dominant_register'] == register]
                    post_reg = post_blast[post_blast['dominant_register'] == register]
                    
                    if len(pre_reg) > 0 and len(post_reg) > 0:
                        sentiment_analysis['register_era_comparison'][register] = {
                            'pre_blast': {
                                'total': len(pre_reg),
                                'avg_score': pre_reg['sentiment_score'].mean(),
                                'positive_pct': len(pre_reg[pre_reg['sentiment'] == 'positive']) / len(pre_reg) * 100,
                                'negative_pct': len(pre_reg[pre_reg['sentiment'] == 'negative']) / len(pre_reg) * 100
                            },
                            'post_blast': {
                                'total': len(post_reg),
                                'avg_score': post_reg['sentiment_score'].mean(),
                                'positive_pct': len(post_reg[post_reg['sentiment'] == 'positive']) / len(post_reg) * 100,
                                'negative_pct': len(post_reg[post_reg['sentiment'] == 'negative']) / len(post_reg) * 100
                            },
                            'sentiment_shift': post_reg['sentiment_score'].mean() - pre_reg['sentiment_score'].mean()
                        }
        
        # Temporal trends based on crawl dates
        print("[STEP] Computing temporal sentiment trends (only valid crawl dates)...")
        if df['crawl_datetime'].notna().sum() > 0:
            df_temporal = df[df['crawl_datetime'].notna()].copy()
            df_temporal['year_month'] = df_temporal['crawl_datetime'].dt.to_period('M')
            temporal_sentiment = df_temporal.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
            
            # Convert Period objects to strings for JSON serialization
            if len(temporal_sentiment) > 0:
                temporal_dict = {}
                for period in temporal_sentiment.index:
                    period_str = str(period)  # Convert Period to string
                    temporal_dict[period_str] = temporal_sentiment.loc[period].to_dict()
                sentiment_analysis['temporal'] = temporal_dict
            else:
                sentiment_analysis['temporal'] = {}
        
        print("[OK] Sentiment analysis complete!")
        return df, keywords, sentiment_analysis
    def create_enhanced_visualizations(self, df, keywords, sentiment_analysis):
        """Create comprehensive visualizations including register analysis"""
        print("[VIZ] Creating comprehensive visualizations...")
        
        # Check font support
        has_indic_font = selected_font and 'DejaVu Sans' not in selected_font
        
        # Determine grid size based on whether register analysis is included
        grid_rows = 4 if self.use_registers and 'by_register' in sentiment_analysis else 3
        grid_cols = 3
        
        fig = plt.figure(figsize=(20, grid_rows * 5))  # Increased height for register plots
        
        # 1. Overall Sentiment Distribution
        plt.subplot(grid_rows, grid_cols, 1)
        if 'overall' in sentiment_analysis and sentiment_analysis['overall']:
            sentiments = list(sentiment_analysis['overall'].keys())
            counts = list(sentiment_analysis['overall'].values())
            colors = {'positive': '#2ECC71', 'neutral': '#F39C12', 'negative': '#E74C3C'}
            bar_colors = [colors.get(s, '#95A5A6') for s in sentiments]
            
            bars = plt.bar(sentiments, counts, color=bar_colors)
            plt.title('Overall Sentiment Distribution\n(Srirampur Mine Blast 2020)', fontsize=14, fontweight='bold')
            plt.ylabel('Number of Documents')
            
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontsize=11)
        
        # 2. Sentiment Comparison: Pre vs Post Blast (only)
        plt.subplot(grid_rows, grid_cols, 2)
        if 'by_era' in sentiment_analysis:
            eras = ['Pre-Blast\n(crawled <Sept 2020)', 'Post-Blast\n(crawled >=Sept 2020)']
            pre_data = sentiment_analysis['by_era']['pre_blast']
            post_data = sentiment_analysis['by_era']['post_blast']
            
            x = np.arange(len(eras))
            width = 0.25
            
            pos_counts = [pre_data['positive'], post_data['positive']]
            neu_counts = [pre_data['neutral'], post_data['neutral']]
            neg_counts = [pre_data['negative'], post_data['negative']]
            
            plt.bar(x - width, pos_counts, width, label='Positive', color='#2ECC71')
            plt.bar(x, neu_counts, width, label='Neutral', color='#F39C12')
            plt.bar(x + width, neg_counts, width, label='Negative', color='#E74C3C')
            
            plt.xlabel('Era (Based on Crawl Date)')
            plt.ylabel('Number of Documents')
            plt.title('Sentiment by Blast Era\n(Only Valid Crawl Dates)', fontsize=14, fontweight='bold')
            plt.xticks(x, eras)
            plt.legend()
            
            # Add totals
            for i, era_data in enumerate([pre_data, post_data]):
                total = era_data['total']
                plt.text(i, max(pos_counts + neu_counts + neg_counts) * 1.1, 
                        f'n={total}', ha='center', va='bottom', fontsize=10)
        
        # 3. Average Sentiment Score Comparison
        plt.subplot(grid_rows, grid_cols, 3)
        if 'by_era' in sentiment_analysis:
            pre_avg = sentiment_analysis['by_era']['pre_blast']['avg_score']
            post_avg = sentiment_analysis['by_era']['post_blast']['avg_score']
            
            eras = ['Pre-Blast\n(Crawl Date)', 'Post-Blast\n(Crawl Date)']
            scores = [pre_avg, post_avg]
            colors = ['#3498DB' if s >= 0 else '#E74C3C' for s in scores]
            
            bars = plt.bar(eras, scores, color=colors)
            plt.title('Average Sentiment Score\n(Srirampur Mine Blast 2020)', fontsize=14, fontweight='bold')
            plt.ylabel('Sentiment Score')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            plt.ylim(-0.5, 0.5)
            
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (0.02 if score >= 0 else -0.05),
                        f'{score:.3f}', ha='center', va='bottom' if score >= 0 else 'top', 
                        fontsize=12, fontweight='bold')
        
        # 4. Top Keywords (Enhanced with Categories)
        plt.subplot(grid_rows, grid_cols, 4)
        if keywords:
            words = list(keywords.keys())[:12]
            counts = list(keywords.values())[:12]
            
            # Handle display for Hindi/Telugu words
            display_words = []
            for word in words:
                if not has_indic_font and any('\u0900' <= char <= '\u097F' for char in word):
                    display_words.append(self.status_tracker.transliterate_hindi(word))
                else:
                    display_words.append(word)
            
            # Enhanced color coding for different keyword categories
            colors = []
            for word in words:
                if any('\u0900' <= char <= '\u097F' for char in word):
                    colors.append('#FF6B6B')  # Red for Hindi
                elif any('\u0C00' <= char <= '\u0C7F' for char in word):
                    colors.append('#FFB347')  # Orange for Telugu
                elif any(term in word.lower() for term in ['rare earth', 'lithium', 'uranium']):
                    colors.append('#9B59B6')  # Purple for rare earth metals
                elif any(term in word.lower() for term in ['environmental', 'pollution', 'contamination']):
                    colors.append('#27AE60')  # Green for environmental
                elif any(term in word.lower() for term in ['safety', 'accident', 'blast', 'explosion']):
                    colors.append('#E74C3C')  # Red for safety/accidents
                elif any(term in word.lower() for term in ['srirampur', 'singerani']):
                    colors.append('#F39C12')  # Orange for Srirampur specific
                else:
                    colors.append('#4ECDC4')  # Teal for other English terms
            
            bars = plt.barh(range(len(display_words)), counts, color=colors)
            plt.yticks(range(len(display_words)), display_words)
            plt.xlabel('Frequency')
            plt.title('Top Mining Keywords by Category\n(Red=Hindi, Orange=Telugu, Purple=Rare Earth)', fontsize=14, fontweight='bold')
            
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_width() + max(counts)*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{count}', ha='left', va='center', fontsize=10)
# 5. Sentiment Percentage Comparison (Pre vs Post Blast only)
        plt.subplot(grid_rows, grid_cols, 5)
        if 'by_era' in sentiment_analysis:
            pre_data = sentiment_analysis['by_era']['pre_blast']
            post_data = sentiment_analysis['by_era']['post_blast']
            
            if pre_data['total'] > 0 and post_data['total'] > 0:
                pre_pct = [pre_data['positive']/pre_data['total']*100, 
                          pre_data['neutral']/pre_data['total']*100, 
                          pre_data['negative']/pre_data['total']*100]
                post_pct = [post_data['positive']/post_data['total']*100, 
                           post_data['neutral']/post_data['total']*100, 
                           post_data['negative']/post_data['total']*100]
                
                x = np.arange(3)
                width = 0.35
                
                sentiment_labels = ['Positive', 'Neutral', 'Negative']
                plt.bar(x - width/2, pre_pct, width, label='Pre-Blast (Crawl)', color='#3498DB', alpha=0.8)
                plt.bar(x + width/2, post_pct, width, label='Post-Blast (Crawl)', color='#E67E22', alpha=0.8)
                
                plt.xlabel('Sentiment Type')
                plt.ylabel('Percentage (%)')
                plt.title('Sentiment Distribution (%)\nPre vs Post Mine Blast', fontsize=14, fontweight='bold')
                plt.xticks(x, sentiment_labels)
                plt.legend()
                
                # Add percentage labels
                for i, (pre, post) in enumerate(zip(pre_pct, post_pct)):
                    plt.text(i - width/2, pre + 1, f'{pre:.1f}%', ha='center', va='bottom', fontsize=9)
                    plt.text(i + width/2, post + 1, f'{post:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 6. Language Content Analysis
        plt.subplot(grid_rows, grid_cols, 6)
        hindi_count = df['has_hindi'].sum()
        telugu_count = df['has_telugu'].sum()
        english_only = len(df) - hindi_count - telugu_count + (df['has_hindi'] & df['has_telugu']).sum()
        mixed = (df['has_hindi'] & df['has_telugu']).sum()
        
        sizes = [hindi_count - mixed, telugu_count - mixed, mixed, english_only]
        labels = ['Hindi Only', 'Telugu Only', 'Mixed Hindi-Telugu', 'English Only']
        colors = ['#FF9999', '#FFB347', '#9999FF', '#66B2FF']
        
        # Filter out zero values
        filtered_sizes = []
        filtered_labels = []
        filtered_colors = []
        for size, label, color in zip(sizes, labels, colors):
            if size > 0:
                filtered_sizes.append(size)
                filtered_labels.append(label)
                filtered_colors.append(color)
        
        plt.pie(filtered_sizes, labels=filtered_labels, autopct='%1.1f%%', colors=filtered_colors, startangle=90)
        plt.title('Language Content Distribution', fontsize=14, fontweight='bold')
        
        # 7. Document Count by Era (Pre vs Post only)
        plt.subplot(grid_rows, grid_cols, 7)
        era_counts = df['blast_era'].value_counts()
        era_labels = {'pre_blast': 'Pre-Blast\n(Crawled <Sept 2020)', 'post_blast': 'Post-Blast\n(Crawled >=Sept 2020)'}
        
        display_labels = [era_labels.get(era, era) for era in era_counts.index]
        colors = {'pre_blast': '#3498DB', 'post_blast': '#E67E22'}
        bar_colors = [colors.get(era, '#95A5A6') for era in era_counts.index]
        
        bars = plt.bar(display_labels, era_counts.values, color=bar_colors)
        plt.title('Document Count by Era\n(Srirampur Mine Blast 2020)', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Documents')
        
        for bar, count in zip(bars, era_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(era_counts.values)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=11)
        
        # 8. Sentiment Confidence Distribution
        plt.subplot(grid_rows, grid_cols, 8)
        plt.hist(df['confidence'], bins=20, alpha=0.7, color='#9B59B6', edgecolor='black')
        plt.xlabel('Sentiment Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Sentiment Analysis Confidence\nDistribution', fontsize=14, fontweight='bold')
        plt.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["confidence"].mean():.3f}')
        plt.legend()
        
        # 9. Register Distribution (if available)
        plt.subplot(grid_rows, grid_cols, 9)
        if self.use_registers and 'dominant_register' in df.columns:
            register_counts = df['dominant_register'].value_counts()
            # Filter out 'unknown' and low-count registers
            filtered_registers = register_counts[register_counts > 3].drop('unknown', errors='ignore')
            
            # Define color palette for register types
            register_colors = {
                'news_report': '#FF6B6B',
                'informational_persuasive': '#4ECDC4',
                'opinion': '#FFB347',
                'interactive_discussion': '#6495ED',
                'legal_formal': '#CCCCFF',
                'academic': '#9B59B6',
                'promotional': '#FFC300',
                'narrative': '#58D68D',
                'how_to_instructional': '#F1948A',
                'lyrical': '#C39BD3',
                'spoken': '#85C1E9'
            }
            
            # Default color for other registers
            default_color = '#95A5A6'
            
            # Create bar colors
            bar_colors = [register_colors.get(reg, default_color) for reg in filtered_registers.index]
            
            # Create horizontal bar chart
            bars = plt.barh(range(len(filtered_registers)), filtered_registers.values, color=bar_colors)
            plt.yticks(range(len(filtered_registers)), 
                      [reg.replace('_', ' ').title() for reg in filtered_registers.index])
            plt.xlabel('Number of Documents')
            plt.title('Document Distribution by Text Register Type', fontsize=14, fontweight='bold')
            
            for i, (bar, count) in enumerate(zip(bars, filtered_registers.values)):
                plt.text(bar.get_width() + max(filtered_registers.values)*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{count}', ha='left', va='center', fontsize=10)
        else:
            plt.axis('off')
            plt.text(0.5, 0.5, 'Register Analysis Not Available', ha='center', va='center',
                    fontsize=14, transform=plt.gca().transAxes)
        
        # If register analysis is enabled, add more plots
        if self.use_registers and 'by_register' in sentiment_analysis and grid_rows > 3:
            # 10. Register-Era Sentiment Comparison
            plt.subplot(grid_rows, grid_cols, 10)
            if 'register_era_comparison' in sentiment_analysis:
                registers = []
                pre_scores = []
                post_scores = []
                sentiment_shifts = []
                
                # Get top registers with sufficient data in both pre and post periods
                for register, data in sentiment_analysis['register_era_comparison'].items():
                    if data['pre_blast']['total'] >= 5 and data['post_blast']['total'] >= 5:
                        registers.append(register.replace('_', ' ').title())
                        pre_scores.append(data['pre_blast']['avg_score'])
                        post_scores.append(data['post_blast']['avg_score'])
                        sentiment_shifts.append(data['sentiment_shift'])
                
                if len(registers) > 0:
                    # Sort by absolute sentiment shift
                    sorted_indices = np.argsort(np.abs(sentiment_shifts))[::-1]
                    registers = [registers[i] for i in sorted_indices[:6]]  # Top 6 for readability
                    pre_scores = [pre_scores[i] for i in sorted_indices[:6]]
                    post_scores = [post_scores[i] for i in sorted_indices[:6]]
                    sentiment_shifts = [sentiment_shifts[i] for i in sorted_indices[:6]]
                    
                    x = np.arange(len(registers))
                    width = 0.35
                    
                    plt.bar(x - width/2, pre_scores, width, label='Pre-Blast', color='#3498DB')
                    plt.bar(x + width/2, post_scores, width, label='Post-Blast', color='#E67E22')
                    
                    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                    plt.xlabel('Register Type')
                    plt.ylabel('Average Sentiment Score')
                    plt.title('Sentiment Score by Register Type\nPre vs Post Blast', fontsize=14, fontweight='bold')
                    plt.xticks(x, registers, rotation=45, ha='right')
                    plt.legend()
                    
                    # Add sentiment shift annotations
                    for i, shift in enumerate(sentiment_shifts):
                        plt.text(i, max(pre_scores + post_scores) * 1.1, 
                                f'Δ{shift:+.2f}', ha='center', va='bottom', 
                                fontsize=10, color='green' if shift > 0 else 'red')
                else:
                    plt.axis('off')
                    plt.text(0.5, 0.5, 'Insufficient Register Data\nfor Comparison', 
                            ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
            else:
                plt.axis('off')
                plt.text(0.5, 0.5, 'Register-Era Comparison\nNot Available', 
                        ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
            
            # 11. News Register Sentiment Distribution
            plt.subplot(grid_rows, grid_cols, 11)
            if 'by_register' in sentiment_analysis and 'news_report' in sentiment_analysis['by_register']:
                news_data = sentiment_analysis['by_register']['news_report']
                
                # Sentiment distribution for news reports
                sentiment_labels = ['Positive', 'Neutral', 'Negative']
                sentiment_counts = [news_data['positive'], news_data['neutral'], news_data['negative']]
                sentiment_colors = ['#2ECC71', '#F39C12', '#E74C3C']
                
                bars = plt.bar(sentiment_labels, sentiment_counts, color=sentiment_colors)
                plt.title('News Reports Sentiment Distribution\n(Journalistic Content)', 
                         fontsize=14, fontweight='bold')
                plt.ylabel('Number of Documents')
                
                for bar, count in zip(bars, sentiment_counts):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sentiment_counts)*0.01,
                            f'{count}', ha='center', va='bottom', fontsize=11)
                    
                plt.annotate(f"Avg Score: {news_data['avg_score']:.3f}", 
                           xy=(0.5, 0.9), xycoords='axes fraction',
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
            else:
                # Handle case where news register isn't available
                plt.axis('off')
                plt.text(0.5, 0.5, 'News Report Register\nNot Available', 
                        ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
            
            # 12. Register Confidence Distribution
            plt.subplot(grid_rows, grid_cols, 12)
            if self.use_registers and 'register_confidence' in df.columns:
                plt.hist(df['register_confidence'], bins=20, alpha=0.7, color='#3498DB', edgecolor='black')
                plt.xlabel('Register Classification Confidence')
                plt.ylabel('Frequency')
                plt.title('Register Classification Confidence\nDistribution', fontsize=14, fontweight='bold')
                plt.axvline(df['register_confidence'].mean(), color='red', linestyle='--',
                           label=f'Mean: {df["register_confidence"].mean():.3f}')
                plt.legend()
            else:
                plt.axis('off')
                plt.text(0.5, 0.5, 'Register Confidence\nNot Available', 
                        ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        
        # Enhanced Summary Statistics - Make this be the last plot
        summary_plot_position = grid_rows * grid_cols
        plt.subplot(grid_rows, grid_cols, summary_plot_position)
        plt.axis('off')
        
        # Calculate key insights
        if 'by_era' in sentiment_analysis:
            pre = sentiment_analysis['by_era']['pre_blast']
            post = sentiment_analysis['by_era']['post_blast']
            
            # Sentiment shift analysis
            pre_pos_pct = pre['positive']/max(1, pre['total'])*100 if pre['total'] > 0 else 0
            post_pos_pct = post['positive']/max(1, post['total'])*100 if post['total'] > 0 else 0
            pos_change = post_pos_pct - pre_pos_pct
            
            pre_neg_pct = pre['negative']/max(1, pre['total'])*100 if pre['total'] > 0 else 0
            post_neg_pct = post['negative']/max(1, post['total'])*100 if post['total'] > 0 else 0
            neg_change = post_neg_pct - pre_neg_pct
            
            score_change = post['avg_score'] - pre['avg_score']
            
            # Register insights
            register_insights = ""
            if self.use_registers and 'register_era_comparison' in sentiment_analysis:
                # Identify register with biggest sentiment shift
                max_shift_register = None
                max_shift_value = 0
                for register, data in sentiment_analysis['register_era_comparison'].items():
                    if abs(data['sentiment_shift']) > abs(max_shift_value):
                        max_shift_value = data['sentiment_shift']
                        max_shift_register = register
                
                if max_shift_register:
                    register_display = max_shift_register.replace('_', ' ').title()
                    shift_direction = "positive" if max_shift_value > 0 else "negative"
                    register_insights = f"\n[REGISTER INSIGHT]\nBiggest sentiment shift: {register_display}\nShift: {max_shift_value:+.3f} (more {shift_direction})"
                    
                    # Add news sentiment if available
                    if 'news_report' in sentiment_analysis['by_register']:
                        news_data = sentiment_analysis['by_register']['news_report']
                        register_insights += f"\nNews reports avg score: {news_data['avg_score']:.3f}"
            
            # Get top mining-related areas/terms
            mining_areas = []
            area_keywords = ['srirampur', 'singerani', 'mancherial', 'telangana', 'sccl', 'mine', 'mining', 'coal']
            for keyword in list(keywords.keys())[:8]:
                if any(area in keyword.lower() or area in keyword for area in area_keywords):
                    if 'srirampur' in keyword.lower() or 'श्रीरामपुर' in keyword:
                        mining_areas.append('Srirampur Mine')
                    elif 'singerani' in keyword.lower() or 'सिंगरेनी' in keyword:
                        mining_areas.append('Singerani Collieries')
                    elif 'mancherial' in keyword.lower() or 'मंचेरियल' in keyword:
                        mining_areas.append('Mancherial District')
                    elif 'telangana' in keyword.lower() or 'तेलंगाना' in keyword:
                        mining_areas.append('Telangana State')
                    elif 'coal' in keyword.lower() or 'कोयला' in keyword:
                        mining_areas.append('Coal Industry')
            
            summary_text = f"""Srirampur Mine Blast 2020 Sentiment Analysis
Mining Discourse Comparison Summary
[ONLY VALID CRAWL DATES - NO UNKNOWNS]

[KEY FINDINGS]

Total Documents (with valid crawl dates): {len(df):,}
- Pre-Blast (crawled <Sept 2, 2020): {pre['total']:,} docs
- Post-Blast (crawled >=Sept 2, 2020): {post['total']:,} docs  
- Crawl Date Coverage: 100% (unknown dates excluded)

[SENTIMENT SHIFTS]

Positive Sentiment:
- Pre-Blast: {pre_pos_pct:.1f}%
- Post-Blast: {post_pos_pct:.1f}%
- Change: {pos_change:+.1f}%

Negative Sentiment:
- Pre-Blast: {pre_neg_pct:.1f}%
- Post-Blast: {post_neg_pct:.1f}%
- Change: {neg_change:+.1f}%

Score Change: {score_change:+.3f}
({'More positive' if score_change > 0.02 else 'More negative' if score_change < -0.02 else 'Similar'})
{register_insights}

[TOP MINING ENTITIES]
{', '.join(set(mining_areas[:3])) if mining_areas else 'Srirampur, SCCL, Coal Mining'}

[DATA QUALITY]
- Hindi Content: {hindi_count}/{len(df)} ({hindi_count/len(df)*100:.1f}%)
- Telugu Content: {telugu_count}/{len(df)}
- Avg Confidence: {df['confidence'].mean():.3f}
- Focus: Mine Safety & Blast Impact
- Source: HPLT Hindi Corpus"""
            
            plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.1))
        
        plt.tight_layout()
        plt.savefig('srirampur_mine_blast_2020_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
       
        print("[OK] Visualizations complete! Saved as srirampur_mine_blast_2020_sentiment_analysis.png")
        
    def save_comprehensive_results(self, df, keywords, sentiment_analysis):
        """Save all analysis results with register information included"""
        print("[SAVE] Saving comprehensive results...")
        
        # Save main dataset with all available fields
        print("[CSV] Saving complete mining dataset...")
        df.to_csv('srirampur_mine_blast_sentiment_data.csv', index=False)
        
        # Save a detailed version with expanded information for manual review
        print("[CSV] Saving detailed mining documents for manual review...")
        detailed_df = df.copy()
        
        # Add keyword categories for easier filtering
        detailed_df['has_rare_earth_keywords'] = detailed_df['keywords_found'].apply(
            lambda x: any(keyword in str(x).lower() for keyword in 
                         ['rare earth', 'lithium', 'uranium', 'ree', 'दुर्लभ पृथ्वी']) if x else False
        )
        
        detailed_df['has_environmental_keywords'] = detailed_df['keywords_found'].apply(
            lambda x: any(keyword in str(x).lower() for keyword in 
                         ['environmental', 'pollution', 'contamination', 'पर्यावरण', 'प्रदूषण']) if x else False
        )
        
        detailed_df['has_safety_keywords'] = detailed_df['keywords_found'].apply(
            lambda x: any(keyword in str(x).lower() for keyword in 
                         ['safety', 'accident', 'blast', 'explosion', 'सुरक्षा', 'दुर्घटना']) if x else False
        )
        
        detailed_df['has_srirampur_keywords'] = detailed_df['keywords_found'].apply(
            lambda x: any(keyword in str(x).lower() for keyword in 
                         ['srirampur', 'singerani', 'श्रीरामपुर', 'सिंगरेनी']) if x else False
        )
        
        # Convert keywords list to string for better CSV readability
        detailed_df['keywords_string'] = detailed_df['keywords_found'].apply(
            lambda x: '; '.join(x) if isinstance(x, list) else str(x)
        )
        
        # Save detailed CSV
        detailed_df.to_csv('srirampur_mine_blast_detailed_analysis.csv', index=False)
        
        # Save separate CSVs for different categories
        print("[CSV] Saving category-specific datasets...")
        
        # Rare Earth Documents
        rare_earth_docs = detailed_df[detailed_df['has_rare_earth_keywords'] == True]
        if len(rare_earth_docs) > 0:
            rare_earth_docs.to_csv('mining_rare_earth_documents.csv', index=False)
            print(f"[CSV] Saved {len(rare_earth_docs)} rare earth mining documents")
        
        # Environmental Impact Documents  
        env_docs = detailed_df[detailed_df['has_environmental_keywords'] == True]
        if len(env_docs) > 0:
            env_docs.to_csv('mining_environmental_impact_documents.csv', index=False)
            print(f"[CSV] Saved {len(env_docs)} environmental impact documents")
        
        # Safety/Accident Documents
        safety_docs = detailed_df[detailed_df['has_safety_keywords'] == True]
        if len(safety_docs) > 0:
            safety_docs.to_csv('mining_safety_accident_documents.csv', index=False)
            print(f"[CSV] Saved {len(safety_docs)} safety/accident documents")
        
        # Srirampur-specific Documents
        srirampur_docs = detailed_df[detailed_df['has_srirampur_keywords'] == True]
        if len(srirampur_docs) > 0:
            srirampur_docs.to_csv('srirampur_specific_documents.csv', index=False)
            print(f"[CSV] Saved {len(srirampur_docs)} Srirampur-specific documents")
        
        # Register-specific documents (if register analysis is enabled)
        if self.use_registers and 'dominant_register' in detailed_df.columns:
            for register in detailed_df['dominant_register'].unique():
                if register != 'unknown':
                    register_docs = detailed_df[detailed_df['dominant_register'] == register]
                    if len(register_docs) >= 5:  # Only save if we have enough documents
                        safe_filename = register.replace(' ', '_').lower()
                        register_docs.to_csv(f'mining_{safe_filename}_documents.csv', index=False)
                        print(f"[CSV] Saved {len(register_docs)} {register} documents")
            
            # Save news documents with sentiment analysis
            if 'news_report' in detailed_df['dominant_register'].values:
                news_docs = detailed_df[detailed_df['dominant_register'] == 'news_report']
                if len(news_docs) > 0:
                    news_docs.to_csv('mining_news_reports.csv', index=False)
                    print(f"[CSV] Saved {len(news_docs)} news report documents")
                    
                    # Split news by era
                    pre_blast_news = news_docs[news_docs['blast_era'] == 'pre_blast']
                    post_blast_news = news_docs[news_docs['blast_era'] == 'post_blast']
                    
                    if len(pre_blast_news) > 0:
                        pre_blast_news.to_csv('mining_pre_blast_news_reports.csv', index=False)
                        print(f"[CSV] Saved {len(pre_blast_news)} pre-blast news reports")
                    
                    if len(post_blast_news) > 0:
                        post_blast_news.to_csv('mining_post_blast_news_reports.csv', index=False)
                        print(f"[CSV] Saved {len(post_blast_news)} post-blast news reports")
        
        # Pre/Post blast comparison CSV
        print("[CSV] Saving temporal comparison data...")
        pre_blast_docs = detailed_df[detailed_df['blast_era'] == 'pre_blast']
        post_blast_docs = detailed_df[detailed_df['blast_era'] == 'post_blast']
        
        if len(pre_blast_docs) > 0:
            pre_blast_docs.to_csv('mining_pre_blast_documents.csv', index=False)
            print(f"[CSV] Saved {len(pre_blast_docs)} pre-blast documents")
        
        if len(post_blast_docs) > 0:
            post_blast_docs.to_csv('mining_post_blast_documents.csv', index=False)
            print(f"[CSV] Saved {len(post_blast_docs)} post-blast documents")
        
        # Save keyword frequency analysis
        print("[CSV] Saving keyword analysis...")
        keyword_df = pd.DataFrame([
            {'keyword': k, 'frequency': v, 'language': 
             'Hindi' if any('\u0900' <= char <= '\u097F' for char in k) else
             'Telugu' if any('\u0C00' <= char <= '\u0C7F' for char in k) else 'English',
             'category': self._categorize_keyword(k)}
            for k, v in keywords.items()
        ])
        keyword_df = keyword_df.sort_values('frequency', ascending=False)
        keyword_df.to_csv('mining_keyword_analysis.csv', index=False)
        
        # Save register analysis (if available)
        if self.use_registers and 'dominant_register' in df.columns:
            print("[CSV] Saving register analysis...")
            register_stats = df.groupby(['dominant_register', 'blast_era']).agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'confidence': ['mean', 'max'],
                'sentiment': lambda x: x.value_counts().to_dict()
            }).reset_index()
            
            # Convert the nested MultiIndex to a flat DataFrame
            register_stats_flat = pd.DataFrame()
            register_stats_flat['register'] = register_stats['dominant_register']
            register_stats_flat['blast_era'] = register_stats['blast_era']
            register_stats_flat['document_count'] = register_stats[('sentiment_score', 'count')]
            register_stats_flat['avg_sentiment'] = register_stats[('sentiment_score', 'mean')]
            register_stats_flat['std_sentiment'] = register_stats[('sentiment_score', 'std')]
            register_stats_flat['avg_confidence'] = register_stats[('confidence', 'mean')]
            register_stats_flat['sentiment_distribution'] = register_stats[('sentiment', '<lambda>')]
            
            # Save register statistics
            register_stats_flat.to_csv('mining_register_analysis.csv', index=False)
            print(f"[CSV] Saved register analysis with {len(register_stats_flat)} register-era combinations")
            
            # Create a register sentiment comparison across eras
            if len(register_stats_flat) > 0:
                register_pivot = register_stats_flat.pivot(index='register', columns='blast_era', 
                                                        values='avg_sentiment')
                
                # Calculate sentiment shift
                if 'pre_blast' in register_pivot.columns and 'post_blast' in register_pivot.columns:
                    register_pivot['sentiment_shift'] = register_pivot['post_blast'] - register_pivot['pre_blast']
                    register_pivot['abs_shift'] = register_pivot['sentiment_shift'].abs()
                    register_pivot.sort_values('abs_shift', ascending=False, inplace=True)
                    register_pivot.drop('abs_shift', axis=1, inplace=True)
                    
                    # Save register sentiment shift analysis
                    register_pivot.to_csv('mining_register_sentiment_shifts.csv')
                    print(f"[CSV] Saved register sentiment shift analysis with {len(register_pivot)} registers")
        
        # Calculate processing stats
        processing_time = time.time() - self.status_tracker.start_time
        
        # Custom JSON serializer to handle various data types
        def json_serializer(obj):
            """Custom JSON serializer for handling pandas objects and other non-serializable types"""
            if pd.isna(obj):
                return None
            elif hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            elif hasattr(obj, '__str__'):  # fallback for other objects
                return str(obj)
            else:
                return obj
        
        # Create comprehensive summary
        summary = {
            'metadata': {
                'analysis_type': 'Srirampur Mine Blast 2020 Sentiment Analysis with HPLT Register Integration',
                'analysis_date': datetime.now().isoformat(),
                'processing_time_seconds': processing_time,
                'data_source': 'HPLT Hindi Corpus',
                'sentiment_analyzer': 'VADER + TextBlob',
                'blast_separator_date': '2020-09-02',
                'focus': 'Srirampur RK 5B Mine Blast - Safety and Impact',
                'location': 'Mancherial district, Telangana, India',
                'company': 'Singerani Collieries Company Limited (SCCL)',
                'temporal_logic': 'Documents crawled before Sept 2, 2020 = Pre-Blast, crawled after = Post-Blast',
                'data_filtering': 'Unknown crawl dates excluded from analysis',
                'register_analysis': self.use_registers,
'improvements': [
                    'Uses crawl date for temporal analysis',
                    'Focus on mining safety and blast impact',
                    'Includes Hindi and Telugu content analysis',
                    'Enhanced keyword matching for mining context',
                    'Regional focus on Telangana mining sector',
                    'Excludes documents with unknown crawl dates',
                    'Includes rare earth metals mining discourse',
                    'Comprehensive environmental impact analysis',
                    'HPLT register analysis for text type classification' if self.use_registers else 'No register analysis'
                ]
            },
            'processing_statistics': {
                'files_processed': self.status_tracker.stats['files_downloaded'],
                'total_documents_scanned': self.status_tracker.stats['docs_processed'],
                'mining_documents_found': self.status_tracker.stats['mining_docs_found'],
                'documents_with_valid_crawl_dates': len(df),
                'success_rate_percentage': (self.status_tracker.stats['mining_docs_found'] / 
                                          max(1, self.status_tracker.stats['docs_processed'])) * 100,
                'processing_speed_docs_per_second': self.status_tracker.stats['docs_processed'] / processing_time,
                'crawl_date_coverage_percentage': 100.0  # Since we filtered out unknowns
            },
            'document_categories': {
                'rare_earth_documents': len(rare_earth_docs) if 'rare_earth_docs' in locals() else 0,
                'environmental_documents': len(env_docs) if 'env_docs' in locals() else 0,
                'safety_accident_documents': len(safety_docs) if 'safety_docs' in locals() else 0,
                'srirampur_specific_documents': len(srirampur_docs) if 'srirampur_docs' in locals() else 0,
                'pre_blast_documents': len(pre_blast_docs) if 'pre_blast_docs' in locals() else 0,
                'post_blast_documents': len(post_blast_docs) if 'post_blast_docs' in locals() else 0
            },
            'temporal_distribution': {
                'pre_blast_documents_by_crawl_date': self.status_tracker.stats['pre_blast_docs'],
                'post_blast_documents_by_crawl_date': self.status_tracker.stats['post_blast_docs'],
                'unknown_crawl_date_documents_excluded': self.status_tracker.stats['mining_docs_found'] - len(df)
            },
            'sentiment_analysis': sentiment_analysis,
            'keyword_analysis': {
                'total_unique_keywords': len(keywords),
                'top_keywords': keywords,
                'mining_specific': [k for k in keywords.keys() 
                                  if any(term in k.lower() for term in ['mine', 'mining', 'coal', 'blast', 'srirampur'])],
                'rare_earth_minerals': [k for k in keywords.keys()
                                      if any(term in k.lower() for term in ['rare earth', 'lithium', 'uranium', 'ree'])],
                'environmental_impact': [k for k in keywords.keys()
                                       if any(term in k.lower() for term in ['pollution', 'contamination', 'environmental', 'ecosystem'])],
                'safety_accidents': [k for k in keywords.keys()
                                   if any(term in k.lower() for term in ['accident', 'safety', 'explosion', 'blast', 'casualt'])],
                'hindi_keywords': [k for k in keywords.keys() 
                                 if any('\u0900' <= char <= '\u097F' for char in k)],
                'telugu_keywords': [k for k in keywords.keys() 
                                  if any('\u0C00' <= char <= '\u0C7F' for char in k)],
                'english_keywords': [k for k in keywords.keys() 
                                   if not any('\u0900' <= char <= '\u0C7F' for char in k)]
            },
            'content_analysis': {
                'total_documents_with_valid_dates': len(df),
                'hindi_content_percentage': float(df['has_hindi'].sum() / len(df) * 100),
                'telugu_content_percentage': float(df['has_telugu'].sum() / len(df) * 100),
                'average_document_length': float(df['text_length'].mean()),
                'average_sentiment_confidence': float(df['confidence'].mean()),
                'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
                'era_distribution_by_crawl_date': df['blast_era'].value_counts().to_dict()
            }
        }
        
        # Add register analysis to summary if available
        if self.use_registers and 'dominant_register' in df.columns:
            # Count documents by register
            register_counts = df['dominant_register'].value_counts().to_dict()
            
            # Calculate register statistics
            summary['register_analysis'] = {
                'documents_with_register_info': self.status_tracker.stats['docs_with_registers'],
                'register_coverage_percentage': self.status_tracker.stats['docs_with_registers'] / max(1, len(df)) * 100,
                'register_distribution': register_counts,
                'average_register_confidence': float(df['register_confidence'].mean()) if 'register_confidence' in df.columns else 0.0,
                'register_sentiment': {}
            }
            
            # Add sentiment by register
            if 'by_register' in sentiment_analysis:
                summary['register_analysis']['register_sentiment'] = sentiment_analysis['by_register']
            
            # Add register-era comparison
            if 'register_era_comparison' in sentiment_analysis:
                summary['register_analysis']['register_era_comparison'] = sentiment_analysis['register_era_comparison']
        
        # Save summary with custom serializer
        try:
            with open('srirampur_mine_blast_sentiment_summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=json_serializer)
        except Exception as e:
            print(f"[WARNING] Error saving summary JSON: {e}")
            # Save as a simplified version if complex structure fails
            simplified_summary = {
                'analysis_type': summary['metadata']['analysis_type'],
                'total_documents': summary['content_analysis']['total_documents_with_valid_dates'],
                'pre_blast_docs': summary['temporal_distribution']['pre_blast_documents_by_crawl_date'],
                'post_blast_docs': summary['temporal_distribution']['post_blast_documents_by_crawl_date'],
                'excluded_unknown_dates': summary['temporal_distribution']['unknown_crawl_date_documents_excluded'],
                'sentiment_distribution': summary['content_analysis']['sentiment_distribution']
            }
            with open('srirampur_mine_blast_sentiment_summary_simple.json', 'w', encoding='utf-8') as f:
                json.dump(simplified_summary, f, indent=2, ensure_ascii=False)
        
        # Save detailed sentiment data by era (only pre_blast and post_blast)
        sentiment_by_era = {}
        for era in ['pre_blast', 'post_blast']:
            era_df = df[df['blast_era'] == era]
            if len(era_df) > 0:
                sentiment_by_era[era] = {
                    'documents': len(era_df),
                    'classification_method': 'crawl_date_based',
                    'sentiment_distribution': era_df['sentiment'].value_counts().to_dict(),
                    'average_score': float(era_df['sentiment_score'].mean()),
                    'score_by_sentiment': {
                        sentiment: float(era_df[era_df['sentiment'] == sentiment]['sentiment_score'].mean())
                        for sentiment in era_df['sentiment'].unique()
                        if len(era_df[era_df['sentiment'] == sentiment]) > 0
                    },
                    'top_keywords': era_df['keywords_found'].explode().value_counts().head(10).to_dict() if len(era_df) > 0 else {},
                    'sample_documents': [
                        {
                            'text': str(doc['text']),
                            'sentiment': str(doc['sentiment']),
                            'sentiment_score': float(doc['sentiment_score']),
                            'keywords_found': doc['keywords_found'],
                            'crawl_date': str(doc['crawl_date']) if doc['crawl_date'] else None,
                            'register': str(doc['dominant_register']) if 'dominant_register' in doc else 'unknown'
                        }
                        for doc in era_df.sample(min(10, len(era_df))).to_dict('records')
                    ] if len(era_df) > 0 else []
                }
                
                # Add register distribution for the era if available
                if self.use_registers and 'dominant_register' in era_df.columns:
                    sentiment_by_era[era]['register_distribution'] = era_df['dominant_register'].value_counts().to_dict()
        
        try:
            with open('sentiment_by_era_mining.json', 'w', encoding='utf-8') as f:
                json.dump(sentiment_by_era, f, indent=2, ensure_ascii=False, default=json_serializer)
        except Exception as e:
            print(f"[WARNING] Error saving era analysis JSON: {e}")
        
        # Save sample documents for verification
        if len(df) > 0:
            try:
                sample_size = min(100, len(df))
                sample_df = df.sample(sample_size)
                sample_df.to_json('mining_document_samples.json', orient='records', 
                                 lines=True, force_ascii=False)
            except Exception as e:
                print(f"[WARNING] Error saving sample documents: {e}")
                
        # If register analysis is enabled, save register analysis details
        if self.use_registers and 'dominant_register' in df.columns:
            # Save register-specific analysis
            register_analysis = {}
            
            for register in df['dominant_register'].unique():
                if register != 'unknown':
                    register_df = df[df['dominant_register'] == register]
                    if len(register_df) > 5:  # Only analyze if we have enough documents
                        register_analysis[register] = {
                            'documents': len(register_df),
                            'sentiment_distribution': register_df['sentiment'].value_counts().to_dict(),
                            'average_score': float(register_df['sentiment_score'].mean()),
                            'pre_post_distribution': register_df['blast_era'].value_counts().to_dict(),
                            'sample_text': str(register_df.sample(1)['text'].values[0][:500]) + '...',
                            'avg_confidence': float(register_df['confidence'].mean()),
                            'top_keywords': register_df['keywords_found'].explode().value_counts().head(5).to_dict() if len(register_df) > 0 else {},
                        }
                        
                        # Compare pre/post blast for this register
                        pre_register = register_df[register_df['blast_era'] == 'pre_blast']
                        post_register = register_df[register_df['blast_era'] == 'post_blast']
                        
                        if len(pre_register) > 0 and len(post_register) > 0:
                            register_analysis[register]['sentiment_shift'] = {
                                'pre_blast_score': float(pre_register['sentiment_score'].mean()),
                                'post_blast_score': float(post_register['sentiment_score'].mean()),
                                'score_change': float(post_register['sentiment_score'].mean() - pre_register['sentiment_score'].mean()),
                                'pre_positive_pct': float(len(pre_register[pre_register['sentiment'] == 'positive']) / len(pre_register) * 100),
                                'post_positive_pct': float(len(post_register[post_register['sentiment'] == 'positive']) / len(post_register) * 100),
                                'positive_pct_change': float(len(post_register[post_register['sentiment'] == 'positive']) / len(post_register) * 100 - 
                                                           len(pre_register[pre_register['sentiment'] == 'positive']) / len(pre_register) * 100)
                            }
            
            try:
                with open('mining_register_analysis_detailed.json', 'w', encoding='utf-8') as f:
                    json.dump(register_analysis, f, indent=2, ensure_ascii=False, default=json_serializer)
                print(f"[JSON] Saved detailed register analysis with {len(register_analysis)} registers")
            except Exception as e:
                print(f"[WARNING] Error saving register analysis JSON: {e}")
        
        print("[OK] Results saved successfully!")
        print("\n[FILES CREATED]")
        print("  [VIZ] srirampur_mine_blast_2020_sentiment_analysis.png - Enhanced visualizations")
        
        print("\n[MAIN CSV FILES]")
        print("  [DATA] srirampur_mine_blast_sentiment_data.csv - Complete dataset")
        print("  [DETAIL] srirampur_mine_blast_detailed_analysis.csv - Enhanced with categories")
        print("  [KEYWORDS] mining_keyword_analysis.csv - Keyword frequency analysis")
        
        if self.use_registers:
            print("  [REGISTER] mining_register_analysis.csv - Register sentiment analysis")
            print("  [REGISTER] mining_register_sentiment_shifts.csv - Register sentiment shifts")
        
        print("\n[CATEGORY-SPECIFIC CSV FILES]")
        if len(rare_earth_docs) > 0:
            print(f"  [RARE-EARTH] mining_rare_earth_documents.csv - {len(rare_earth_docs)} documents")
        if len(env_docs) > 0:
            print(f"  [ENVIRONMENT] mining_environmental_impact_documents.csv - {len(env_docs)} documents")
        if len(safety_docs) > 0:
            print(f"  [SAFETY] mining_safety_accident_documents.csv - {len(safety_docs)} documents")
        if len(srirampur_docs) > 0:
            print(f"  [SRIRAMPUR] srirampur_specific_documents.csv - {len(srirampur_docs)} documents")
        
        print("\n[TEMPORAL CSV FILES]")
        if len(pre_blast_docs) > 0:
            print(f"  [PRE-BLAST] mining_pre_blast_documents.csv - {len(pre_blast_docs)} documents")
        if len(post_blast_docs) > 0:
            print(f"  [POST-BLAST] mining_post_blast_documents.csv - {len(post_blast_docs)} documents")
        
        if self.use_registers and 'dominant_register' in df.columns:
            print("\n[REGISTER-SPECIFIC CSV FILES]")
            for register in df['dominant_register'].unique():
                if register != 'unknown':
                    register_df = df[df['dominant_register'] == register]
                    if len(register_df) >= 5:
                        safe_name = register.replace(' ', '_').lower()
                        print(f"  [{register.upper()}] mining_{safe_name}_documents.csv - {len(register_df)} documents")
        
        print("\n[JSON FILES]")
        print("  [SUMMARY] srirampur_mine_blast_sentiment_summary.json - Analysis summary")
        print("  [DETAILS] sentiment_by_era_mining.json - Era comparison details")
        print("  [SAMPLES] mining_document_samples.json - Document samples")
        if self.use_registers:
            print("  [REGISTER] mining_register_analysis_detailed.json - Register analysis details")
    
    def _categorize_keyword(self, keyword):
       """Categorize a keyword based on its content"""
       keyword_lower = keyword.lower()
       
       # Check for rare earth metals
       if any(term in keyword_lower for term in ['rare earth', 'lithium', 'uranium', 'ree', 'दुर्लभ पृथ्वी']):
           return 'Rare Earth Metals'
       
       # Check for environmental terms
       elif any(term in keyword_lower for term in ['environmental', 'pollution', 'contamination', 'पर्यावरण', 'प्रदूषण']):
           return 'Environmental Impact'
       
       # Check for safety/accident terms
       elif any(term in keyword_lower for term in ['safety', 'accident', 'blast', 'explosion', 'सुरक्षा', 'दुर्घटना']):
           return 'Safety & Accidents'
       
       # Check for Srirampur specific
       elif any(term in keyword_lower for term in ['srirampur', 'singerani', 'श्रीरामपुर', 'सिंगरेनी']):
           return 'Srirampur Specific'
       
       # Check for general mining
       elif any(term in keyword_lower for term in ['mine', 'mining', 'coal', 'खनन', 'कोयला']):
           return 'General Mining'
       
       # Check for corporate/legal
       elif any(term in keyword_lower for term in ['corporate', 'legal', 'law', 'policy', 'कानून', 'नीति']):
           return 'Corporate & Legal'
       
       else:
           return 'Other'

def main():
    """Main function to run Srirampur Mine Blast 2020 sentiment analysis with HPLT register integration"""
    print("[TITLE] SRIRAMPUR MINE BLAST 2020 SENTIMENT ANALYZER WITH HPLT REGISTER INTEGRATION")
    print("="*80)
    print("[PURPOSE] Analyze mining discourse sentiment")
    print("[COMPARISON] Before vs After Srirampur Mine Blast 2020")
    print("[TEMPORAL] Uses CRAWL DATE for temporal analysis")
    print("[LOGIC] Documents crawled before Sept 2, 2020 = Pre-Blast")
    print("[LOGIC] Documents crawled after Sept 2, 2020 = Post-Blast")
    print("[FILTER] Excludes documents with unknown crawl dates")
    print("[FOCUS] Mining safety, underground operations, blast impact")
    print("[LOCATION] Mancherial district, Telangana, India")
    print("[METHOD] VADER + TextBlob sentiment analysis")
    print("[SOURCE] HPLT Hindi Corpus")
    print("[REGISTERS] Optional HPLT register analysis for document types")
    print("="*80)
   
    # Configuration
    print("\n[CONFIG] Analysis Configuration:")
    print("1. Quick Analysis (5 files, 15k docs each, ~8 min)")
    print("2. Standard Analysis (8 files, 25k docs each, ~20 min)")
    print("3. Comprehensive Analysis (12 files, 35k docs each, ~35 min)")
    print("4. Custom Configuration")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    # Ask about register analysis
    use_registers = input("\nInclude HPLT register analysis? (y/n): ").strip().lower() == 'y'
    print(f"[REGISTER] {'Using' if use_registers else 'Not using'} HPLT register analysis")
    
    if choice == "1":
        analyzer = SrirampurMineBlastSentimentAnalyzer(max_files=5, max_docs_per_file=15000, 
                                                     download_chunks_mb=12, use_registers=use_registers)
        print("[MODE] Quick Analysis Selected")
    elif choice == "2":
        analyzer = SrirampurMineBlastSentimentAnalyzer(max_files=8, max_docs_per_file=25000, 
                                                     download_chunks_mb=15, use_registers=use_registers)
        print("[MODE] Standard Analysis Selected")
    elif choice == "3":
        analyzer = SrirampurMineBlastSentimentAnalyzer(max_files=12, max_docs_per_file=35000, 
                                                     download_chunks_mb=18, use_registers=use_registers)
        print("[MODE] Comprehensive Analysis Selected")
    elif choice == "4":
        try:
            max_files = int(input("Number of files to process (1-20): "))
            max_docs = int(input("Max documents per file (10000-50000): "))
            chunk_mb = int(input("Download chunk size in MB (10-25): "))
            analyzer = SrirampurMineBlastSentimentAnalyzer(max_files=max_files, max_docs_per_file=max_docs, 
                                                         download_chunks_mb=chunk_mb, use_registers=use_registers)
            print("[MODE] Custom Configuration Applied")
        except:
            print("[ERROR] Invalid input. Using standard configuration.")
            analyzer = SrirampurMineBlastSentimentAnalyzer(max_files=8, max_docs_per_file=25000, 
                                                         download_chunks_mb=15, use_registers=use_registers)
    else:
        print("[MODE] Using Standard Analysis (default)")
        analyzer = SrirampurMineBlastSentimentAnalyzer(max_files=8, max_docs_per_file=25000, 
                                                     download_chunks_mb=15, use_registers=use_registers)
    
    start_time = time.time()
    
    # Step 1: Download and extract mining documents
    print(f"\n[STEP 1] Starting document extraction...")
    print("[INFO] Live status updates below:")
    # Change this line to use the correct method name
    docs = analyzer.download_and_analyze()  # This should match the method name in the class
    
    if not docs:
        print("\n[ERROR] No mining documents found!")
        print("[SUGGESTION] Check network connection or try different configuration")
        return
    
    print(f"\n[SUCCESS] Found {len(docs)} mining documents total")
    
    # Step 2: Analyze sentiment trends
    print("\n" + "="*80)
    print("[STEP 2] Analyzing sentiment trends...")
    df, keywords, sentiment_analysis = analyzer.analyze_sentiment_trends(docs)
    
    if df is None or len(df) == 0:
        print("\n[ERROR] No documents with valid crawl dates found!")
        print("[SUGGESTION] Try processing more files or different configuration")
        return
    
    # Step 3: Create visualizations
    print("\n" + "="*80)
    print("[STEP 3] Creating enhanced visualizations...")
    analyzer.create_enhanced_visualizations(df, keywords, sentiment_analysis)
    
    # Step 4: Save results
    print("\n" + "="*80)
    print("[STEP 4] Saving comprehensive results...")
    analyzer.save_comprehensive_results(df, keywords, sentiment_analysis)
    
    # Final summary with key insights
    total_time = time.time() - start_time
    print(f"\n[SUCCESS] ANALYSIS COMPLETED!")
    print("="*80)
    print(f"[TIME] Total Time: {total_time:.1f} seconds")
    print(f"[DOCS] Mining Documents Found: {len(df):,}")
    print(f"[FILTER] Excluded documents with unknown crawl dates")
    print(f"[COVERAGE] 100% of analyzed documents have valid crawl dates")
    
    if sentiment_analysis and 'by_era' in sentiment_analysis:
        pre = sentiment_analysis['by_era']['pre_blast']
        post = sentiment_analysis['by_era']['post_blast']
        
        print(f"[PRE-BLAST] {pre['total']} docs (crawled before Sept 2, 2020)")
        print(f"    Pos: {pre['positive']}, Neu: {pre['neutral']}, Neg: {pre['negative']}")
        print(f"[POST-BLAST] {post['total']} docs (crawled after Sept 2, 2020)")
        print(f"    Pos: {post['positive']}, Neu: {post['neutral']}, Neg: {post['negative']}")
        
        if pre['total'] > 0 and post['total'] > 0:
            pre_pos_pct = pre['positive']/pre['total']*100
            post_pos_pct = post['positive']/post['total']*100
            change = post_pos_pct - pre_pos_pct
            
            print(f"\n[KEY INSIGHT] Positive sentiment:")
            print(f"  Pre-Blast: {pre_pos_pct:.1f}% | Post-Blast: {post_pos_pct:.1f}%")
            print(f"  Change: {change:+.1f}% ({'Increase' if change > 0 else 'Decrease'})")
            
            score_change = post['avg_score'] - pre['avg_score']
            print(f"[KEY INSIGHT] Average sentiment score change: {score_change:+.3f}")
            print(f"  ({'Mining discourse became more positive' if score_change > 0.02 else 'Mining discourse became more negative' if score_change < -0.02 else 'Similar sentiment in mining discourse'})")
    
    # Print register insights if available
    if use_registers and 'register_era_comparison' in sentiment_analysis:
        print("\n[REGISTER INSIGHTS]")
        
        # Find register with biggest sentiment shift
        max_shift_register = None
        max_shift_value = 0
        for register, data in sentiment_analysis['register_era_comparison'].items():
            if abs(data['sentiment_shift']) > abs(max_shift_value):
                max_shift_value = data['sentiment_shift']
                max_shift_register = register
        
        if max_shift_register:
            reg_display = max_shift_register.replace('_', ' ').title()
            shift_direction = "positive" if max_shift_value > 0 else "negative"
            print(f"  Biggest sentiment shift: {reg_display}")
            print(f"  Shift direction: {shift_direction.upper()}")
            print(f"  Shift magnitude: {abs(max_shift_value):.3f}")
            
            # Add news sentiment if available
            if 'news_report' in sentiment_analysis['by_register']:
                news_data = sentiment_analysis['by_register']['news_report']
                news_sentiment = "positive" if news_data['avg_score'] > 0.05 else "negative" if news_data['avg_score'] < -0.05 else "neutral"
                print(f"  News articles sentiment: {news_sentiment.upper()} ({news_data['avg_score']:.3f})")
                
                # Print news distribution
                total_news = news_data['total']
                pos_pct = news_data['positive'] / total_news * 100
                neg_pct = news_data['negative'] / total_news * 100
                neu_pct = news_data['neutral'] / total_news * 100
                print(f"  News distribution: {pos_pct:.1f}% positive, {neu_pct:.1f}% neutral, {neg_pct:.1f}% negative")
    
    print(f"\n[FILES] Check current directory for analysis results")
    print("[MAIN] srirampur_mine_blast_2020_sentiment_analysis.png")
    print("[DATA] srirampur_mine_blast_sentiment_data.csv")
    print("[SUMMARY] srirampur_mine_blast_sentiment_summary.json")
    
    if use_registers:
        print("[REGISTER] mining_register_analysis.csv")
        print("[REGISTER] mining_register_sentiment_shifts.csv")
        print("[REGISTER] mining_register_analysis_detailed.json")
    
    print("="*80)


if __name__ == "__main__":
    main()
