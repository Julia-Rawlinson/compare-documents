import os
import docx
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_pdf(file_path):
    pdf_text = ""
    with open(file_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            pdf_text += page.extract_text()
    return pdf_text

def compare_pair(our_doc_path, their_doc_path):
    our_text = read_docx(our_doc_path)
    their_text = read_pdf(their_doc_path)

    vectorizer = TfidfVectorizer().fit_transform([our_text, their_text])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)[0][1]

    return os.path.basename(our_doc_path), os.path.basename(their_doc_path), similarity

def compare_documents(our_docs_folder, their_docs_folder):
    our_docs = [f for f in os.listdir(our_docs_folder) if f.endswith('.docx')]
    their_docs = [f for f in os.listdir(their_docs_folder) if f.endswith('.pdf')]

    data = []
    total_comparisons = len(our_docs) * len(their_docs)
    current_comparison = 0

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(compare_pair, os.path.join(our_docs_folder, our_doc), os.path.join(their_docs_folder, their_doc))
            for our_doc in our_docs for their_doc in their_docs
        ]
        for future in as_completed(futures):
            our_doc, their_doc, similarity = future.result()
            data.append({
                'Our Document': our_doc,
                'Their Document': their_doc,
                'Similarity': similarity
            })
            current_comparison += 1
            print(f"Compared {current_comparison} of {total_comparisons} document pairs")

    data = sorted(data, key=lambda x: x['Similarity'], reverse=True)
    df = pd.DataFrame(data)
    df.to_excel('document_similarity_report.xlsx', index=False)
    print("Document similarity report saved as 'document_similarity_report.xlsx'")

if __name__ == '__main__':
    our_docs_folder = '/Users/juliarawlinson/Desktop/unitydocuments'
    their_docs_folder = '/Users/juliarawlinson/Desktop/lldocuments'
    compare_documents(our_docs_folder, their_docs_folder)



