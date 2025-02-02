import os
import json
import unittest
from collections import defaultdict
from rougescore import rouge_n
from pythonrouge.pythonrouge import Pythonrouge
from sumeval.metrics.rouge import RougeCalculator


class TestRougeJA(unittest.TestCase):

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data/rouge")

    def load_test_data(self):
        test_file = os.path.join(self.DATA_DIR, "ROUGE-test-ja.json")
        with open(test_file, encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _split(self, text):
        _txt = text.replace("。", " ").replace("、", " ").strip()
        words = _txt.split(" ")
        words = [w.strip() for w in words if w.strip()]
        return words

    def _compress(self, text_or_texts):
        if isinstance(text_or_texts, (tuple, list)):
            return ["".join(s.split(" ")) for s in text_or_texts]
        else:
            return "".join(text_or_texts.split(" "))

    def test_rouge(self):
        data = self.load_test_data()
        rouge = RougeCalculator(stopwords=False, lang="ja")
        for eval_id in data:
            summaries = data[eval_id]["summaries"]
            references = data[eval_id]["references"]
            for n in [1, 2]:
                for s in summaries:
                    v = rouge.rouge_n(self._compress(s),
                                      self._compress(references), n)
                    b_v = rouge_n(self._split(s),
                                  [self._split(r) for r in references],
                                  n, 0.5)
                    self.assertLess(abs(b_v - v), 1e-5)

    def test_rouge_with_stop_words(self):
        data = self.load_test_data()
        rouge = RougeCalculator(stopwords=True, lang="ja")

        def split(text):
            words = self._split(text)
            words = [w for w in words if not rouge._lang.is_stop_word(w)]
            return words

        for eval_id in data:
            summaries = data[eval_id]["summaries"]
            references = data[eval_id]["references"]
            for n in [1, 2]:
                for s in summaries:
                    v = rouge.rouge_n(s, references, n)
                    b_v = rouge_n(split(s),
                                  [split(r) for r in references],
                                  n, 0.5)
                    self.assertLess(abs(b_v - v), 1e-5)

    def test_summary_rouge_l(self):
        def word2ids(summary, reference):
            id_dict = defaultdict(lambda: len(id_dict))
            summary = [" ".join([str(id_dict[w]) for w in sent.split()]) for sent in summary]
            reference = [" ".join([str(id_dict[w]) for w in sent.split()]) for sent in reference]
            return summary, reference

        data = self.load_test_data()
        rouge = RougeCalculator(stopwords=False, lang="ja", split_summaries=True)
        sentence_tokenizer = rouge.get_sentence_tokenizer()
        for eval_id in data:
            summaries = data[eval_id]["summaries"]
            references = data[eval_id]["references"]

            pythonrouge_summaries = [" ".join(rouge.tokenize(s)) for s in sentence_tokenizer("".join(summaries))]
            pythonrouge_references = [" ".join(rouge.tokenize(s)) for s in sentence_tokenizer("".join(references))]

            pythonrouge_summaries, pythonrouge_references = word2ids(pythonrouge_summaries, pythonrouge_references)
            baseline = Pythonrouge(summary_file_exist=False,
                                   summary=[pythonrouge_summaries],
                                   reference=[[pythonrouge_references]],
                                   n_gram=1, recall_only=False, ROUGE_L=True,
                                   length_limit=False,
                                   stemming=False, stopwords=False)
            pythonrouge_score = baseline.calc_score()

            sumeval_summary = "".join(self._compress(summaries))
            sumeval_reference = "".join(self._compress(references))
            rouge_l_score = rouge.rouge_l_summary(sumeval_summary, sumeval_reference)

            self.assertLess(abs(pythonrouge_score["ROUGE-L-F"] - rouge_l_score), 1e-5)


if __name__ == "__main__":
    unittest.main()
