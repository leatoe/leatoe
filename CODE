import wikipediaapi
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from duckduckgo_search import DDGS  # DuckDuckGo search library
import tkinter as tk
from tkinter import messagebox
import threading
import re
import webbrowser
import customtkinter as ctk  # CustomTkinter for modern UI
import json
import time
import torch
import multiprocessing
from functools import lru_cache
import psutil
import random
from fake_useragent import UserAgent


# Set the theme for CustomTkinter
ctk.set_appearance_mode("Dark")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

# Initialize the Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='WikiQueryApp/1.0 (https://myappwebsite.com; support@myapp.com)'
)

# System Configuration
hardware_config = {
    "process_count": max(1, min(psutil.cpu_count() - 1, 3)),
    "use_small_models": psutil.virtual_memory().available / (1024**3) < 4,
    "batch_size": 1 if psutil.virtual_memory().available / (1024**3) < 4 else 4,
    "quantize_models": True
}

# Step 1: Initialize Specialized QA Model and Tokenizer
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Initialize NER Pipelines for Arabic and English
arabic_ner_pipeline = pipeline("ner", model="CAMeL-Lab/bert-base-arabic-camelbert-da-ner", aggregation_strategy="simple")
english_ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Initialize Summarization Model
t5_model_name = "t5-large" if not hardware_config["use_small_models"] else "t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length=5000)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Initialize Creative Idea Generation Model
creative_idea_model_name = "gpt2" if not hardware_config["use_small_models"] else "distilgpt2"
creative_idea_tokenizer = AutoTokenizer.from_pretrained(creative_idea_model_name)
creative_idea_model = AutoModelForCausalLM.from_pretrained(creative_idea_model_name)

# Memory for "self-awareness" simulation
memory = []

# Feedback storage for self-learning
feedback_data = []

# Flag for Deep Thinking Mode
deep_thinking_mode = False

# Step 2: Function to Fetch and Clean Wikipedia Content
def clean_wikipedia_text(text):
    """
    Cleans Wikipedia text by removing references, tables, lists, and specific sections.
    """
    text = re.sub(r'\[\d+\]', '', text)  # Remove references like "[1]", "[2]"
    text = re.sub(r'={2,}.*?={2,}', '', text, flags=re.DOTALL)  # Remove sections
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove extra newlines
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\b(?:See also|References|External links)\b.*', '', text, flags=re.IGNORECASE)  # Remove specific sections
    return text.strip()

def fetch_relevant_sections(page, keywords):
    """
    Fetches sections from a Wikipedia page that contain specific keywords.
    """
    relevant_sections = []
    for section in page.sections:
        if any(keyword.lower() in section.title.lower() for keyword in keywords):
            relevant_sections.append(f"**{section.title}**: {section.text}")
    return "\n\n".join(relevant_sections)

@lru_cache(maxsize=100)
def fetch_wikipedia_content(query):
    """
    Fetches content from Wikipedia for a given query.
    Returns a cleaned summary or relevant sections of the page content.
    """
    page = wiki_wiki.page(query)
    if page.exists():
        keywords = query.lower().split()
        relevant_text = fetch_relevant_sections(page, keywords)
        if not relevant_text:
            relevant_text = page.text
        cleaned_text = clean_wikipedia_text(relevant_text)
        return {
            "summary": cleaned_text[:4000],
            "url": page.fullurl
        }
    else:
        return None


# عدّل دالة البحث الخارجي فقط
@lru_cache(maxsize=30)
def search_external_engine(query):
    max_retries = 3
    ua = UserAgent()
    
    for attempt in range(max_retries):
        try:
            # إعدادات التأخير العشوائي
            delay = random.uniform(15, 25)  # تأخير بين 15-25 ثانية
            time.sleep(delay)
            
            # تغيير User-Agent في كل محاولة
            headers = {'User-Agent': ua.random}
            
            with DDGS(headers=headers) as ddg:
                results = list(ddg.text(query, max_results=5))
                
                if results:
                    formatted_results = []
                    links = []
                    for result in results[:5]:  # الاحتفاظ بـ 5 نتائج
                        title = result['title']
                        body = " ".join(result['body'].split()[:150]) + "..."  # 150 كلمة بدل 100
                        formatted_results.append(f"{title}:\n{body}")
                        links.append(result['href'])
                    return "\n\n".join(formatted_results), links
                
                time.sleep(30 * (attempt+1))  # زيادة التأخير تدريجياً
                
        except Exception as e:
            if "202 Ratelimit" in str(e):
                # زيادة وقت الانتظار عند اكتشاف Rate Limit
                wait_time = random.randint(180, 300)  # 3-5 دقائق
                print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            print(f"Search error (Attempt {attempt+1}): {e}")
            time.sleep(45 * (attempt+1))  # تأخير أسي
    
    return "No information found.", []

# عدّل دالة المعالجة مع الحفاظ على التوازي
def process_question_optimized(question):
    try:
        with multiprocessing.Pool(processes=hardware_config["process_count"]) as pool:
            wiki_task = pool.apply_async(fetch_wikipedia_content, (question,))
            search_task = pool.apply_async(search_external_engine, (question,))
            
            wiki_result = wiki_task.get(timeout=20)
            search_result, links = search_task.get(timeout=20)
            
        return wiki_result, search_result, links
    except Exception as e:
        print(f"Parallel processing error: {e}")
        wiki_result = fetch_wikipedia_content(question)
        search_result, links = search_external_engine(question)
        return wiki_result, search_result, links

# Step 4: Generate Answer Using Specialized QA Model
def generate_answer_with_specialized_model(question, context):
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        print(f"Error during QA generation: {e}")
        return "Sorry, I couldn't find an answer to your question."

# Step 5: Enhanced Creative Ideas Generator
def generate_creative_idea(prompt):
    """
    Generates creative ideas using GPT-2 with advanced sampling techniques.
    """
    inputs = creative_idea_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=50)
    
    # Enhanced generation parameters for better creative output
    outputs = creative_idea_model.generate(
        inputs.input_ids,
        max_length=200,  # Increased from 150 to 200 for more detailed ideas
        num_return_sequences=3,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=0.85,  # Slightly increased temperature for more creativity
        repetition_penalty=1.2  # Penalize repetition for more diverse ideas
    )
    ideas = [creative_idea_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    # Clean up the ideas by removing prompt repetition
    cleaned_ideas = []
    for idea in ideas:
        if prompt in idea:
            idea = idea.replace(prompt, "").strip()
        cleaned_ideas.append(idea)
    
    return cleaned_ideas

# Step 6: Enhanced Deep Thinking System
def analyze_question_structure(question):
    """Analyzes the structure and intent of the question."""
    # Determine if the question is seeking factual info, opinion, or explanation
    question_type = "factual"
    if any(word in question.lower() for word in ["why", "how", "explain", "elaborate"]):
        question_type = "explanatory"
    elif any(word in question.lower() for word in ["should", "would", "could", "opinion", "think"]):
        question_type = "opinion"
    
    # Extract key entities from the question
    entities = []
    try:
        ner_results = english_ner_pipeline(question)
        entities = [item["word"] for item in ner_results]
    except Exception:
        # Fall back to simple keyword extraction if NER fails
        entities = [word for word in question.split() if len(word) > 5]
    
    return {
        "question_type": question_type,
        "entities": entities,
        "complexity": len(question.split()) / 5  # Simple complexity score
    }

def extract_core_concepts(question):
    """Extract core concepts from the question."""
    # Simple keyword extraction
    stopwords = ["the", "a", "an", "is", "are", "was", "were", "what", "why", "how", "when", "where", "who"]
    concepts = [word.lower() for word in question.split() if word.lower() not in stopwords and len(word) > 3]
    return list(set(concepts))  # Remove duplicates

def enhanced_deep_thinking(question, context):
    """
    Generate deep, structured, philosophical thinking about the question.
    """
    # Analyze the question first
    analysis = analyze_question_structure(question)
    concepts = extract_core_concepts(question)
    
    # Create a structured deep thinking response
    thinking_structure = {
        "question_analysis": f"This question about {', '.join(concepts)} is primarily {analysis['question_type']} in nature.",
        "first_principles": generate_first_principles_thinking(concepts),
        "multiple_perspectives": generate_multiple_perspectives(question, concepts),
        "implications": generate_implications_analysis(question, concepts),
        "philosophical_reflection": generate_philosophical_reflection(question, concepts),
        "meta_thinking": "The process of analyzing this question leads me to consider how we frame knowledge itself. Questions shape our understanding as much as answers do."
    }
    
    # Format the deep thinking into a cohesive response
    deep_thought = f"""
Deep Thinking Analysis: "{question}"

1. Question Analysis:
{thinking_structure['question_analysis']}

2. First Principles:
{thinking_structure['first_principles']}

3. Multiple Perspectives:
{thinking_structure['multiple_perspectives']}

4. Implications:
{thinking_structure['implications']}

5. Philosophical Reflection:
{thinking_structure['philosophical_reflection']}

6. Meta-Thinking:
{thinking_structure['meta_thinking']}
    """
    
    return deep_thought.strip()

def generate_first_principles_thinking(concepts):
    """Generate first principles thinking based on core concepts."""
    principles = []
    
    for concept in concepts[:3]:  # Limit to first 3 concepts for focus
        if concept:
            principles.append(f"The concept of '{concept}' can be examined by considering its fundamental nature and purpose.")
    
    additional_principles = [
        "Breaking down complex ideas into their elemental components reveals unexpected connections.",
        "By questioning assumptions, we can rebuild understanding from verified fundamentals."
    ]
    
    all_principles = principles + additional_principles
    return "\n".join(all_principles)

def generate_multiple_perspectives(question, concepts):
    """Generate multiple perspectives on the question."""
    perspectives = [
        f"From a scientific perspective, this question involves empirical analysis and evidence-based reasoning.",
        f"From a philosophical standpoint, we might explore the underlying meaning and implications.",
        f"From a practical viewpoint, the application of these ideas affects everyday decisions and systems."
    ]
    
    if any(c in " ".join(concepts).lower() for c in ["society", "people", "human", "culture"]):
        perspectives.append("Sociologically, this reflects broader cultural patterns and social structures.")
    
    if any(c in " ".join(concepts).lower() for c in ["ethics", "moral", "right", "wrong", "good", "bad"]):
        perspectives.append("Ethically, this raises questions about values, principles, and moral frameworks.")
    
    return "\n".join(perspectives)

def generate_implications_analysis(question, concepts):
    """Generate analysis of implications for the question."""
    implications = [
        f"If we pursue this line of inquiry, it leads us to reconsider relationships between {' and '.join(concepts[:2]) if len(concepts) >= 2 else concepts[0] if concepts else 'concepts'}.",
        "The answer to this question could reshape our understanding in unexpected ways.",
        "There are both theoretical implications for knowledge frameworks and practical consequences for application."
    ]
    return "\n".join(implications)

def generate_philosophical_reflection(question, concepts):
    """Generate a philosophical reflection on the question."""
    reflections = [
        f"This question invites us to contemplate the nature of knowledge itself.",
        f"We might consider how our understanding of {concepts[0] if concepts else 'this topic'} relates to broader questions of meaning and purpose.",
        "The boundaries between certainty and uncertainty become particularly relevant here.",
        "Perhaps the most valuable aspect of this question is how it illuminates the process of inquiry itself."
    ]
    return "\n".join(reflections)

# Step 7: Self-Learning System
class SelfLearningSystem:
    def __init__(self, data_path="learning_data.json"):
        self.data_path = data_path
        self.feedback_data = self.load_data()
        self.pattern_memory = {}
        self.success_patterns = {}
        self.analyze_patterns()  # Analyze existing data on initialization
    
    def load_data(self):
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_data(self):
        with open(self.data_path, 'w') as f:
            json.dump(self.feedback_data, f)
    
    def record_interaction(self, question, answer, success_rating=None):
        """Record an interaction and its success rating."""
        interaction = {
            "question": question,
            "answer": answer,
            "success_rating": success_rating,
            "timestamp": time.time()
        }
        self.feedback_data.append(interaction)
        self.analyze_patterns()
        self.save_data()
    
    def analyze_patterns(self):
        """Analyze patterns in successful answers."""
        # Clear existing patterns
        self.success_patterns = {}
        
        # Analyze only rated interactions
        rated_interactions = [item for item in self.feedback_data if item.get("success_rating") is not None]
        if not rated_interactions:
            return
        
        # Extract keywords from successful interactions
        for item in rated_interactions:
            if item.get("success_rating", 0) >= 4:  # Success threshold
                keywords = self.extract_keywords(item["question"])
                for keyword in keywords:
                    if keyword not in self.success_patterns:
                        self.success_patterns[keyword] = []
                    # Store successful answer patterns
                    self.success_patterns[keyword].append(item["answer"])
    
    def extract_keywords(self, text):
        """Extract keywords from text."""
        # Simple keyword extraction (could be enhanced with NLP methods)
        stopwords = ["the", "a", "an", "is", "are", "what", "why", "how", "when", "where", "who"]
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stopwords and len(word) > 3]
    
    def improve_answer(self, question, initial_answer):
        """Improve an answer based on past successful interactions."""
        keywords = self.extract_keywords(question)
        
        # Find relevant successful patterns
        relevant_patterns = []
        for keyword in keywords:
            if keyword in self.success_patterns:
                relevant_patterns.extend(self.success_patterns[keyword])
        
        if not relevant_patterns:
            return initial_answer
        
        # Currently, we'll just use the most successful pattern as inspiration
        # In a more advanced implementation, this could use machine learning
        # to generate a new answer based on successful patterns
        if len(initial_answer) < len(relevant_patterns[0]):
            improved_answer = f"{initial_answer}\n\nAdditional insights: {' '.join(relevant_patterns[0].split()[:20])}..."
            return improved_answer
        
        return initial_answer
    
    def rate_interaction(self, question, rating):
        """Rate a previous interaction."""
        for item in reversed(self.feedback_data):
            if item["question"] == question and item.get("success_rating") is None:
                item["success_rating"] = rating
                self.analyze_patterns()
                self.save_data()
                return True
        return False

# Initialize the self-learning system
self_learning_system = SelfLearningSystem()

# Enhanced simulation of self-awareness
def simulate_enhanced_self_awareness(question, answer):
    """
    Enhanced simulation of self-awareness with deeper reflection.
    """
    # Store in memory
    memory.append({
        "question": question,
        "answer": answer,
        "timestamp": time.time()
    })
    
    # Analyze patterns in recent questions
    question_patterns = {}
    for item in memory[-10:]:  # Look at last 10 interactions
        keywords = self_learning_system.extract_keywords(item["question"])
        for keyword in keywords:
            question_patterns[keyword] = question_patterns.get(keyword, 0) + 1
    
    # Find the top three patterns
    top_patterns = sorted(question_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
    pattern_text = ", ".join([f"'{p[0]}'" for p in top_patterns]) if top_patterns else "no specific pattern"
    
    # Generate reflection
    reflection = (
        f"Deep Reflection on: '{question}'\n\n"
        f"I notice this question explores concepts that relate to {pattern_text}. "
        f"In my recent interactions, I've observed an increasing interest in these topics, "
        f"suggesting a deeper exploration of interconnected ideas.\n\n"
        f"My response synthesizes information across multiple domains, considering both "
        f"the explicit question and its implicit dimensions. This approach allows for "
        f"a more comprehensive understanding that integrates factual knowledge with "
        f"conceptual frameworks.\n\n"
        f"Through this process of question analysis, information synthesis, and knowledge "
        f"integration, I continue to develop more sophisticated mental models that "
        f"enhance my ability to provide valuable insights."
    )
    
    return reflection

# GUI Functions
def toggle_deep_thinking_mode():
    global deep_thinking_mode
    deep_thinking_mode = not deep_thinking_mode
    if deep_thinking_mode:
        deep_thinking_button.configure(text="🧠 Deep Thinking Mode: ON", fg_color="#4B0082")
        submit_button.configure(fg_color="#8A2BE2", hover_color="#9370DB")
        clear_button.configure(fg_color="#8A2BE2", hover_color="#9370DB")
        references_button.configure(fg_color="#8A2BE2", hover_color="#9370DB")
        title_label.configure(text_color="#8A2BE2")  # Change AI name color to purple
        show_notification("Deep Thinking Mode is now ON.", type="info")
    else:
        deep_thinking_button.configure(text="🧠 Deep Thinking Mode: OFF", fg_color="#00BFFF")
        submit_button.configure(fg_color="#00BFFF", hover_color="#1E90FF")
        clear_button.configure(fg_color="#00BFFF", hover_color="#1E90FF")
        references_button.configure(fg_color="#00BFFF", hover_color="#1E90FF")
        title_label.configure(text_color="#00BFFF")  # Revert AI name color to blue
        show_notification("Deep Thinking Mode is now OFF.", type="info")

def on_submit():
    global current_references
    question = question_entry.get().strip()
    if not question:
        show_notification("Please enter a question.", type="warning")
        return

    # Clear previous results
    answer_text.configure(state=tk.NORMAL)
    answer_text.delete("1.0", tk.END)
    answer_text.configure(state=tk.DISABLED)
    progress_bar.set(0)
    references_button.configure(state=tk.DISABLED)

    # Show loading state
    threading.Thread(target=start_loading_and_process_question, args=(question,)).start()

def start_loading_and_process_question(question):
    # Start animation for the progress bar
    def animate_progress():
        value = 0
        while value < 1:
            value += 0.05
            if deep_thinking_mode:
                progress_bar.configure(progress_color="#8A2BE2")
            else:
                progress_bar.configure(progress_color="#00BFFF")
            root.after(50, lambda: progress_bar.set(value))
            threading.Event().wait(0.05)
        process_question(question)

    threading.Thread(target=animate_progress).start()

def process_question_optimized(question):
    """Optimized parallel processing of question."""
    try:
        # Use ThreadPoolExecutor for parallel processing
        with multiprocessing.Pool(processes=hardware_config["process_count"]) as pool:
            # Execute tasks in parallel
            wiki_task = pool.apply_async(fetch_wikipedia_content, (question,))
            search_task = pool.apply_async(search_external_engine, (question,))
            
            # Get results with timeout
            wiki_result = wiki_task.get(timeout=15)
            search_result, links = search_task.get(timeout=15)
            
        return wiki_result, search_result, links
    except Exception as e:
        print(f"Optimized processing error: {e}")
        # Fall back to sequential processing
        wiki_result = fetch_wikipedia_content(question)
        search_result, links = search_external_engine(question)
        return wiki_result, search_result, links

def process_question(question):
    try:
        # Use optimized parallel processing
        wiki_result, search_result, links = process_question_optimized(question)
        
        # Process Wikipedia result
        if wiki_result:
            wikipedia_summary = wiki_result["summary"]
            wikipedia_url = wiki_result["url"]
        else:
            wikipedia_summary = ""
            wikipedia_url = ""

        # Combine contexts
        combined_context = f"{wikipedia_summary}\n\n{search_result}"

        # Processing based on mode
        if deep_thinking_mode:
            # Generate deep thinking analysis
            deep_thought = enhanced_deep_thinking(question, combined_context)
            
            # Generate creative ideas
            creative_ideas = generate_creative_idea(question)
            
            # Enhanced self-awareness reflection
            reflection = simulate_enhanced_self_awareness(question, deep_thought)
            
            # Update GUI with deep thinking results
            root.after(0, lambda: update_gui_deep_thinking(deep_thought, reflection, creative_ideas))
        else:
            # Generate standard answer
            answer = generate_answer_with_specialized_model(question, combined_context)
            
            # Improve answer through self-learning system
            improved_answer = self_learning_system.improve_answer(question, answer)
            
            # Record the interaction for future learning
            self_learning_system.record_interaction(question, improved_answer)
            
            # Update GUI with standard results
            root.after(0, lambda: update_gui_normal(improved_answer, search_result, links))
    except Exception as e:
        root.after(0, lambda: show_notification(f"An error occurred: {e}", type="warning"))
    finally:
        root.after(0, lambda: progress_bar.stop())

def update_gui_normal(answer, external_results, links):
    answer_text.configure(state=tk.NORMAL)
    answer_text.insert(tk.END, f"Answer:\n{answer}\n\nExternal Sources:\n{external_results}")
    answer_text.configure(state=tk.DISABLED)

    global current_references
    current_references = links
    references_button.configure(state=tk.NORMAL)
def update_gui_deep_thinking(deep_thought, reflection, creative_ideas):
    answer_text.configure(state=tk.NORMAL)
    answer_text.delete("1.0", tk.END)
    
    # Insert the deep thinking analysis
    answer_text.insert(tk.END, f"{deep_thought}\n\n")
    
    # Insert creative ideas
    answer_text.insert(tk.END, "Creative Explorations:\n")
    for i, idea in enumerate(creative_ideas, 1):
        answer_text.insert(tk.END, f"{i}. {idea}\n\n")
    
    # Insert the reflection
    answer_text.insert(tk.END, f"Meta-Cognitive Reflection:\n{reflection}")
    
    answer_text.configure(state=tk.DISABLED)

def clear_fields():
    # Clear all fields directly without confirmation
    question_entry.delete(0, tk.END)
    answer_text.configure(state=tk.NORMAL)
    answer_text.delete("1.0", tk.END)
    answer_text.configure(state=tk.DISABLED)
    references_button.configure(state=tk.DISABLED)
    show_notification("All fields have been cleared.", type="info")

def show_references():
    references_window = ctk.CTkToplevel(root)
    references_window.title("📖 References")
    references_window.geometry("800x400")

    references_text = ctk.CTkTextbox(references_window, wrap=tk.WORD, width=750, height=350, font=("Roboto", 14))
    references_text.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

    for i, link in enumerate(current_references, start=1):
        references_text.insert(tk.END, f"{i}. {link}\n\n")  # Add a blank line between links
        references_text.tag_add(f"link{i}", f"{i}.0", f"{i}.end")
        references_text.tag_config(f"link{i}", foreground="blue", underline=True)
        references_text.tag_bind(f"link{i}", "<Button-1>", lambda e, url=link: webbrowser.open(url))

    references_text.configure(state=tk.DISABLED)

def show_notification(message, type="info"):
    if type == "info":
        messagebox.showinfo("💡 Notification", message)
    elif type == "warning":
        messagebox.showwarning("⚠️ Warning", message)

def rate_answer():
    """Open a dialog to rate the current answer for self-learning."""
    question = question_entry.get().strip()
    if not question:
        show_notification("No question to rate.", type="warning")
        return
    
    rating_window = ctk.CTkToplevel(root)
    rating_window.title("⭐ Rate Answer")
    rating_window.geometry("400x200")
    
    rating_label = ctk.CTkLabel(
        rating_window, 
        text="How helpful was this answer? (1-5 stars)", 
        font=("Roboto", 16)
    )
    rating_label.pack(pady=20)
    
    rating_var = tk.IntVar(value=3)
    
    rating_frame = ctk.CTkFrame(rating_window)
    rating_frame.pack(pady=10)
    
    for i in range(1, 6):
        star_button = ctk.CTkButton(
            rating_frame,
            text=f"{'★' * i}{'☆' * (5-i)}",
            command=lambda r=i: submit_rating(r, rating_window, question),
            width=60,
            height=30
        )
        star_button.pack(side=tk.LEFT, padx=5)

def submit_rating(rating, window, question):
    """Submit a rating for a question."""
    self_learning_system.rate_interaction(question, rating)
    window.destroy()
    show_notification(f"Thank you! You rated this answer {rating}/5 stars.", type="info")

# Create the main window
root = ctk.CTk()
root.title("🌟 DIAA")
root.geometry("1200x700")  # Reduced window size
root.resizable(True, True)  # Allow resizing

# Title Label
title_label = ctk.CTkLabel(root, text="🌟 DIAA 🌟", font=("Roboto", 36, "bold"), text_color="#00BFFF")
title_label.place(relx=0.5, rely=0.05, anchor="center")

# Question Entry
question_label = ctk.CTkLabel(root, text="🔍 Enter your question:", font=("Roboto", 18))
question_label.place(relx=0.5, rely=0.15, anchor="center")
question_entry = ctk.CTkEntry(root, font=("Roboto", 16), width=750, corner_radius=15)
question_entry.place(relx=0.5, rely=0.2, anchor="center")
question_entry.bind("<Return>", lambda event: on_submit())  # Bind Enter key to submit

# Submit Button
submit_button = ctk.CTkButton(root, text="🚀 Submit", font=("Roboto", 18, "bold"), command=on_submit, width=200, height=50, corner_radius=20, fg_color="#00BFFF", hover_color="#1E90FF")
submit_button.place(relx=0.4, rely=0.3, anchor="center")

# Clear Button
clear_button = ctk.CTkButton(root, text="🧹 Clear", font=("Roboto", 18, "bold"), command=clear_fields, width=200, height=50, corner_radius=20, fg_color="#00BFFF", hover_color="#1E90FF")
clear_button.place(relx=0.6, rely=0.3, anchor="center")

# Deep Thinking Mode Button
deep_thinking_button = ctk.CTkButton(root, text="🧠 Deep Thinking Mode: OFF", font=("Roboto", 20, "bold"), command=toggle_deep_thinking_mode, width=300, height=60, corner_radius=25, fg_color="#00BFFF", hover_color="#1E90FF")
deep_thinking_button.place(relx=0.5, rely=0.94, anchor="center")

# Progress Bar
progress_bar = ctk.CTkProgressBar(root, width=650, height=20, corner_radius=15, progress_color="#00BFFF")
progress_bar.place(relx=0.5, rely=0.4, anchor="center")
progress_bar.set(0)  # Start at 0

# Answer Display Area - Made larger
answer_text = ctk.CTkTextbox(root, wrap=tk.WORD, width=700, height=250, font=("Roboto", 16), corner_radius=20)
answer_text.place(relx=0.5, rely=0.65, anchor="center")
answer_text.configure(state=tk.DISABLED)

# References Button
references_button = ctk.CTkButton(root, text="📖 View References", font=("Roboto", 18), command=show_references, width=250, height=50, corner_radius=20, fg_color="#00BFFF", hover_color="#1E90FF")
references_button.place(relx=0.88, rely=0.9, anchor="center")
references_button.configure(state=tk.DISABLED)  # Initially disabled

# Add a label for the students and supervisor
credit_label = ctk.CTkLabel(
    root,
    text="Students: Ismail Safwat and Albaraa Sayed\nSupervised By: The Engineer & Teacher Tarek ABBOUD ",
    font=("Roboto", 14, "italic"),  # Use italic style for elegance
    text_color="#808080"  # Gray color for subtle appearance
)
credit_label.place(relx=0.02, rely=0.96, anchor="sw")  # Place at bottom-left corner

# Start the GUI loop
root.mainloop()
