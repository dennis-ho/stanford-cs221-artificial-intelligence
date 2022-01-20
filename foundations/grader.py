#!/usr/bin/env python3

import collections
import graderUtil
import random

grader = graderUtil.Grader()
submission = grader.load('submission')

############################################################
# Problems 1 and 2

# Problem 1
grader.add_manual_part('1a', max_points=2, description='optimize weighted average')
grader.add_manual_part('1b', max_points=3, description='swap sum and max')
grader.add_manual_part('1c', max_points=3, description='expected value of iterated game')
grader.add_manual_part('1d', max_points=3, description='derive maximum likelihood')
grader.add_manual_part('1e', max_points=3, description='manipulate conditional probabilities')
grader.add_manual_part('1f', max_points=4, description='take gradient')

# Problem 2
grader.add_manual_part('2a', max_points=2, description='counting faces')
grader.add_manual_part('2b', max_points=3, description='dynamic program')

############################################################
# Problem 3a: findAlphabeticallyLastWord

grader.add_basic_part('3a-0-basic', lambda:
                      grader.require_is_equal('word', submission.find_alphabetically_last_word(
                        'which is the last word alphabetically')),
                      description='simple test case')

grader.add_basic_part('3a-1-basic',
                      lambda: grader.require_is_equal('sun', submission.find_alphabetically_last_word('cat sun dog')),
                      description='simple test case')
grader.add_basic_part('3a-2-basic', lambda: grader.require_is_equal('99999', submission.find_alphabetically_last_word(
    ' '.join(str(x) for x in range(100000)))), description='big test case')

############################################################
# Problem 3b: euclideanDistance

grader.add_basic_part('3b-0-basic', lambda: grader.require_is_equal(5, submission.euclidean_distance((1, 5), (4, 1))),
                      description='simple test case')


def test():
    random.seed(42)
    for _ in range(100):
        x1 = random.randint(0, 10)
        y1 = random.randint(0, 10)
        x2 = random.randint(0, 10)
        y2 = random.randint(0, 10)
        ans2 = submission.euclidean_distance((x1, y1), (x2, y2))


grader.add_hidden_part('3b-1-hidden', test, max_points=2, description='100 random trials')


############################################################
# Problem 3c: mutateSentences

def test():
    grader.require_is_equal(sorted(['a a a a a']), sorted(submission.mutate_sentences('a a a a a')))
    grader.require_is_equal(sorted(['the cat']), sorted(submission.mutate_sentences('the cat')))
    grader.require_is_equal(
        sorted(['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']),
        sorted(submission.mutate_sentences('the cat and the mouse')))


grader.add_basic_part('3c-0-basic', test, max_points=1, description='simple test')


def gen_sentence(alphabet_size, length):
    return ' '.join(str(random.randint(0, alphabet_size)) for _ in range(length))


def test():
    random.seed(42)
    for _ in range(10):
        sentence = gen_sentence(3, 5)
        ans2 = submission.mutate_sentences(sentence)


grader.add_hidden_part('3c-1-hidden', test, max_points=2, description='random trials')


def test():
    random.seed(42)
    for _ in range(10):
        sentence = gen_sentence(25, 10)
        ans2 = submission.mutate_sentences(sentence)


grader.add_hidden_part('3c-2-hidden', test, max_points=3, description='random trials (bigger)')


############################################################
# Problem 3d: dotProduct

def test():
    grader.require_is_equal(15, submission.sparse_vector_dot_product(collections.defaultdict(float, {'a': 5}),
                                                                     collections.defaultdict(float, {'b': 2, 'a': 3})))


grader.add_basic_part('3d-0-basic', test, max_points=1, description='simple test')


def randvec():
    v = collections.defaultdict(float)
    for _ in range(10):
        v[random.randint(0, 10)] = random.randint(0, 10) - 5
    return v


def test():
    random.seed(42)
    for _ in range(10):
        v1 = randvec()
        v2 = randvec()
        ans2 = submission.sparse_vector_dot_product(v1, v2)


grader.add_hidden_part('3d-1-hidden', test, max_points=3, description='random trials')


############################################################
# Problem 3e: incrementSparseVector

def test():
    v = collections.defaultdict(float, {'a': 5})
    submission.increment_sparse_vector(v, 2, collections.defaultdict(float, {'b': 2, 'a': 3}))
    grader.require_is_equal(collections.defaultdict(float, {'a': 11, 'b': 4}), v)


grader.add_basic_part('3e-0-basic', test, description='simple test')


def test():
    random.seed(42)
    for _ in range(10):
        v1a = randvec()
        v1b = v1a.copy()
        v2 = randvec()
        submission.increment_sparse_vector(v1b, 4, v2)
        for key in list(v1b):
            if v1b[key] == 0:
                del v1b[key]


grader.add_hidden_part('3e-1-hidden', test, max_points=3, description='random trials')


############################################################
# Problem 3f: findSingletonWords

def test3f():
    grader.require_is_equal({'quick', 'brown', 'jumps', 'over', 'lazy'},
                            submission.find_singleton_words('the quick brown fox jumps over the lazy fox'))


grader.add_basic_part('3f-0-basic', test3f, description='simple test')


def test3f(num_tokens, num_types):
    import random
    random.seed(42)
    text = ' '.join(str(random.randint(0, num_types)) for _ in range(num_tokens))
    ans2 = submission.find_singleton_words(text)


grader.add_hidden_part('3f-1-hidden', lambda: test3f(1000, 10), max_points=1, description='random trials')
grader.add_hidden_part('3f-2-hidden', lambda: test3f(10000, 100), max_points=2, description='random trials (bigger)')

grader.grade()
