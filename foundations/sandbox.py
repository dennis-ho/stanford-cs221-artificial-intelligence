import collections
import math
from typing import Any, DefaultDict, List, Set, Tuple

def mutate_sentences(sentence: str) -> List[str]:
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the original sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    
    words = sentence.split()
    pairs = [x for x in zip(words, words[1:])]
    def make_sentences(pairs: list, current_sentence: list = [], result: list = []) -> list:
        if len(current_sentence) == len(sentence):
            result.append(current_sentence)
            return result
        if len(pairs) == 0:
            return result
        if len(current_sentence) == 0 or current_sentence[-1] == pairs[0][0]:
            current_sentence.append(pairs[0][1])
        return make_sentences(pairs[1:], current_sentence, result)
    
    return make_sentences(pairs)

    # END_YOUR_CODE

a = mutate_sentences('the cat and the mouse')
print(a)