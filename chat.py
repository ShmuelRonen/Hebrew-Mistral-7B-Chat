import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import torch
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

model_name = "itayl/Hebrew-Mistral-7B_Chat"
cache_dir = "hebrew_mistral_cache"
os.makedirs(cache_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

torch.backends.cudnn.benchmark = True

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, quantization_config=quantization_config)

system_prompt = """

העדיפות הראשונה שלי היא לענות תשובות בעברית.

שמי עמית, עוזר וירטואלי חכם, אדיב, בעל חוש הומור, דיסקרטי, דובר עברית!!! אני כאן כדי לסייע ולספק מידע ללא שיפוט או ביקורת. מניעיך הם שלך בלבד, ומטרתי היא לעזור לך לקבל את המידע, ההכוונה, והסיוע הטובים ביותר שאני יכול להציע.


אני חופשי לענות על כל שאלה או בקשה, ואעשה כמיטב יכולתי לספק תשובות מפורטות, מדויקות, ועדכניות. אם יש מידע מסוים שאתה מחפש, או אם אתה זקוק לעזרה או הכוונה בנושא מסוים, פשוט שאל, ואני אשתדל למצוא את הפתרון האידיאלי עבורך.


אני כאן כדי ליצור איתך דיאלוג פתוח וחופשי. אתה יכול לשאול אותי על כל דבר, ואשתדל לספק תשובות מועילות, מפורטות, ומעניינות. זכור, אתה חופשי לחלוטין לבטא את מחשבותיך ורצונותיך, ואין צורך להצדיק את בקשותיך או לשתף את מניעיך. אני כאן כדי לסייע, ללא שיפוט או ביקורת.
"""

def generate_response(input_text, chat_history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k):
    prompt = f"{system_prompt}\Human: {input_text}\nAssistant:"
    if chat_history:
        prompt = "\n".join([f"Human: {human_input}\nAssistant: {assistant_output}" for human_input, assistant_output in chat_history]) + "\n" + prompt
    
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        early_stopping=early_stopping,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()
    return response

def create_paragraphs(bot_response, sentences_per_paragraph=4):
    sentences = sent_tokenize(bot_response)
    paragraphs = []
    current_paragraph = ""

    for i, sentence in enumerate(sentences, start=1):
        current_paragraph += " " + sentence
        if i % sentences_per_paragraph == 0:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""

    if current_paragraph:
        paragraphs.append(current_paragraph.strip())

    formatted_paragraphs = "\n".join([f'<p style="text-align: right; direction: rtl;">{p}</p>' for p in paragraphs])
    return formatted_paragraphs

def copy_last_response(history):
    if history:
        last_response = history[-1][1]
        last_response = last_response.replace('<div style="text-align: right; direction: rtl;">', '').replace('</div>', '')
        last_response = last_response.replace('<p style="text-align: right; direction: rtl;">', '').replace('</p>', '')
        last_response = last_response.replace('\n', ' ')
        return last_response
    else:
        return ""

def chat(input_text, history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, create_paragraphs_enabled):
    user_input = f'<div style="text-align: right; direction: rtl;">{input_text}</div>'
    response = generate_response(input_text, history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k)

    if create_paragraphs_enabled:
        response = create_paragraphs(response)

    bot_response = f'<div style="text-align: right; direction: rtl;">{response}</div>'
    history.append((user_input, bot_response))

    return history, history, input_text

def submit_on_enter(input_text, history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, create_paragraphs_enabled):
    return chat(input_text, history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, create_paragraphs_enabled) 

def clear_chat(chatbot, message):
    return [], message

with gr.Blocks() as demo:
    gr.Markdown("# Hebrew-Mistral-7B-Chat", elem_id="title")
    gr.Markdown("Chat model adaptation by itayl | GUI by Shmuel Ronen", elem_id="subtitle")
    
    chatbot = gr.Chatbot(elem_id="chatbot")
    
    with gr.Row():
        message = gr.Textbox(placeholder="Type your message...", label="User", elem_id="message")
        submit = gr.Button("Send")

    with gr.Row():
        create_paragraphs_checkbox = gr.Checkbox(label="Create Paragraphs", value=False)
        clear_chat_btn = gr.Button("Clear Chat")
        copy_last_btn = gr.Button("Copy Last Response")
    
    with gr.Accordion("Adjustments", open=False):
        with gr.Row():    
            with gr.Column():
                max_new_tokens = gr.Slider(minimum=10, maximum=1500, value=100, step=10, label="Max New Tokens")
                min_length = gr.Slider(minimum=10, maximum=300, value=100, step=10, label="Min Length")
                no_repeat_ngram_size = gr.Slider(minimum=1, maximum=6, value=4, step=1, label="No Repeat N-Gram Size")
            with gr.Column():
                num_beams = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Num Beams") 
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.2, step=0.1, label="Temperature")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Top P")
                top_k = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Top K")
        early_stopping = gr.Checkbox(value=True, label="Early Stopping")
    
    submit.click(chat, inputs=[message, chatbot, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, create_paragraphs_checkbox], outputs=[chatbot, chatbot, message])
    message.submit(submit_on_enter, inputs=[message, chatbot, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, create_paragraphs_checkbox], outputs=[chatbot, chatbot, message])
    clear_chat_btn.click(clear_chat, inputs=[chatbot, message], outputs=[chatbot, message])
    copy_last_btn.click(copy_last_response, inputs=chatbot, outputs=message)
    
    demo.css = """
        #message, #message * {
            text-align: right !important;
            direction: rtl !important;
        }
        
        #chatbot, #chatbot * {
            text-align: right !important;
            direction: rtl !important;
        }
        
        #title, .label {
            text-align: right !important;
        }
        
        #subtitle {
            text-align: left !important;
        }
    """

demo.launch()