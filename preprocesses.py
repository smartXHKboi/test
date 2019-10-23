import re
import nltk
# Remove noise from text and lemmatize
def remove_noise_text(X_input):
    documents = []
    for sen in range(0, len(X_input)):
        document = re.sub(r'\\r', ' ', str(X_input[sen]))
        document = re.sub(r'\\n', ' ', document)
        document = re.sub(r'(?:^| )\w(?:$| )', ' ', document)
        document = re.sub(r'\W', ' ', document)
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        document = re.sub(r"[^A-Za-z0-9^,!.'+-=]", " ", document)
        document = re.sub(r"what\'s", "what is ", document)
        document = re.sub(r"\'s", " ", document)
        document = re.sub(r"\@", " ", document)
        document = re.sub(r"\"", " ", document)
        document = re.sub(r"\'ve", " have ", document)
        document = re.sub(r"n\'t", " not ", document)
        document = re.sub(r"i\'m", "i am ", document)
        document = re.sub(r"\'re", " are ", document)
        document = re.sub(r"\'d", " would ", document)
        document = re.sub(r"\'ll", " will ", document)
        document = re.sub(r"\,", " ", document)
        document = re.sub(r"\.", " ", document)
        document = re.sub(r"\!", " ! ", document)
        document = re.sub(r"\/", " ", document)
        document = re.sub(r"\^", " ^ ", document)
        document = re.sub(r"\+", " + ", document)
        document = re.sub(r"\-", " - ", document)
        document = re.sub(r"\=", " = ", document)
        document = re.sub(r"\'", " ", document)
        document = re.sub(r"(\d+)(k)", r"\g<1>000", document)
        document = re.sub(r":", " : ", document)
        document = re.sub(r" e g ", " eg ", document)
        document = re.sub(r" b g ", " bg ", document)
        document = re.sub(r" u s ", " american ", document)
        document = re.sub(r"\0s", "0", document)
        document = re.sub(r" 9 11 ", "911", document)
        document = re.sub(r"e - mail", "email", document)
        document = re.sub(r"j k", "jk", document)
        document = re.sub(r"\s{2,}", " ", document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        from nltk.stem import WordNetLemmatizer
        stemmer = WordNetLemmatizer()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    return documents
# Remove stop words and only use nouns and verbs.
def remove_stop_words(corpus):
    result = []
    is_nv = lambda pos: (pos[:2] == 'NN' or pos[:2] == 'VBP')

    for text in corpus:
        tmp = text.split(' ')

        result.append(" ".join([word for (word, pos) in nltk.pos_tag(tmp) if is_nv(pos)]))
    return result
