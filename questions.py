import nltk
import sys
# importing other modules
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Initialize dictionary d
    d = {}
    # Get all of the text files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                # Read files into the dictionary
                f = open(directory+os.sep+file, encoding="utf-8")
                text = f.read()
                f.close()
                d[file] = text
    # Return dictionary
    return d


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Filter out punctuation and convert string to lowercase
    new_doc = ""
    for c in document:
        if c not in string.punctuation:
            new_doc += c         
    new_doc = new_doc.lower()
    # Create list of words
    old_l = nltk.tokenize.word_tokenize(new_doc)
    # Filter out stopwords
    new_l = []
    for word in old_l:
        if word not in nltk.corpus.stopwords.words("english"):
            new_l.append(word)
    # Return list of words
    return new_l


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Create list of lists
    lol = []
    for v in documents.values():
        lol.append(v)
    # Initialize new dictionary
    new_dict = {}
    # Iterate through all words
    total = len(lol)
    for sl in lol:
        for s in sl:
            if s not in new_dict:
                # Add entry to the new dictionary
                count = 0
                for i in range(total):
                    if s in lol[i]:
                        count += 1
                new_dict[s] = math.log(total/count)
    return new_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Initialize dictionary mapping each file to its tf_idf score
    score = {}
    words = list(query)
    for f in files.keys():
        # Compute the dot product of the tf-idf value of each word
        dp = []
        for w in words:
            # Compute the tf value for each word
            tf = 0
            for check in files[f]:
                if w == check:
                    tf += 1
            # Multiply tf and idf and append it to the list
            try:
                dp.append(tf*idfs[w])
            except:
                pass
        score[f] = sum(dp)
    # Create and return list of length n of the highest ranked files
    top_f = []
    for i in range(n):
        best = max(score, key=score.get)
        top_f.append(best)
        del score[best]
    return top_f


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    l_1 = []
    l_2 = []
    l_3 = []
    query_words = list(query)
    for sen, sen_words in sentences.items():
        mwm = 0
        top = 0
        bot = len(sen_words)
        for word in query_words:
            if word in sen_words:
                top += 1
                mwm += idfs[word]
        l_1.append(sen)
        l_2.append(mwm)
        l_3.append(top/bot)
    # Return a list of length n
    ret_l = []
    for i in range(n):
        # get max mwm
        mx_mwm = max(l_2)
        if l_2.count(mx_mwm) > 1:
            # tf tiebraker
            sub_l = []
            i_l = []
            for j in range(len(l_1)):
                if l_2[j] == mx_mwm:
                    sub_l.append(l_3[j])
                    i_l.append(j)
            mx_td = max(sub_l)
            for k in i_l:
                if l_3[k] == mx_td:
                    ret_l.append(l_1[k])
        else:
            ret_l.append(l_1[l_2.index(mx_mwm)])
    return ret_l


if __name__ == "__main__":
    main()
