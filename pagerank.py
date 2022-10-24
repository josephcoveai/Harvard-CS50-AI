import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    new_dict = {}
    # Create dictionary with 1-d probability
    dmp = (1-damping_factor) / len(corpus)
    for k in corpus.keys():
        new_dict[k] = dmp
    # Add damping factor probability if there are no links
    if len(corpus[page]) == 0:
        d = damping_factor / len(corpus)
        for v in corpus.values():
            v += d
        return new_dict
    # Add damping factor probability given links
    d = damping_factor / len(corpus[page])
    for s in corpus[page]:
        new_dict[s] += d
    return new_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Populate new_dict and new_list
    new_dict = {}
    new_list = []
    for k in corpus.keys():
        new_dict[k] = 0
        new_list.append(k)
    # Run the sampling simulation
    v = new_list[random.randrange(len(new_list))]
    new_dict[v] += 1
    for i in range(n-1):
        k_list = []
        val_list = []
        for k, val in transition_model(corpus, v, damping_factor).items():
            k_list.append(k)
            val_list.append(val)
        r = random.random()
        j = 0
        while r > val_list[j]:
            r -= val_list[j]
            j += 1
        v = k_list[j]
        new_dict[v] += 1
    # Divide by n
    for k in new_dict.keys():
        new_dict[k] = new_dict[k] / n
    return new_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initializing dictionaries
    old_dict = {}
    new_dict = {}
    l = len(corpus)
    for k in corpus.keys():
        old_dict[k] = 1 / l
    # Looping through pagerank algorithm
    ep = False
    while ep == False:
        # Populating new_dict
        for k in old_dict.keys():
            link_list = []
            for kk, v in corpus.items():
                for i in v:
                    if i == k:
                        link_list.append(kk)
            # Summation aspect of equation
            summa = 0
            for link in link_list:
                ll = len(corpus[link])
                if ll == 0:
                    ll = l
                summa += (old_dict[link] / ll)
            # Populating new_dict with pagerank equation
            new_dict[k] = ((1 - damping_factor) / l) + (damping_factor * summa)
        # Checking epsilon values
        ep = True
        for k in new_dict.keys():
            if abs(new_dict[k] - old_dict[k]) > .001:
                ep = False
        # Reset dictionaries
        old_dict = {}
        for k in new_dict.keys():
            old_dict[k] = new_dict[k]
        new_dict = {}
    # Return converged dictionary
    return old_dict


if __name__ == "__main__":
    main()
