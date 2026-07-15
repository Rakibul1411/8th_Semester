from collections import Counter
class Yahtzee:
  
    def __init__(self, d1, d2, d3, d4, _5):
        self.dice = [d1, d2, d3, d4, _5]

    @staticmethod
    def _values(d1, d2, d3, d4, d5):
        return [d1, d2, d3, d4, d5]

    @staticmethod
    def _score_face(face, dice):
        return face * dice.count(face)

    @staticmethod
    def _counts(dice):
        return Counter(dice)

    @staticmethod
    def chance(d1, d2, d3, d4, d5):
        return sum(Yahtzee._values(d1, d2, d3, d4, d5))

    @staticmethod
    def yahtzee(dice):
        return 50 if len(set(dice)) == 1 else 0

    @staticmethod
    def ones(d1, d2, d3, d4, d5):
        return Yahtzee._score_face(1, Yahtzee._values(d1, d2, d3, d4, d5))

    @staticmethod
    def twos(d1, d2, d3, d4, d5):
        return Yahtzee._score_face(2, Yahtzee._values(d1, d2, d3, d4, d5))

    @staticmethod
    def threes(d1, d2, d3, d4, d5):
        return Yahtzee._score_face(3, Yahtzee._values(d1, d2, d3, d4, d5))

    def fours(self):
        return Yahtzee._score_face(4, self.dice)

    def fives(self):
        return Yahtzee._score_face(5, self.dice)

    def sixes(self):
        return Yahtzee._score_face(6, self.dice)

    @staticmethod
    def score_pair(d1, d2, d3, d4, d5):
        counts = Yahtzee._counts(Yahtzee._values(d1, d2, d3, d4, d5))
        pairs = [face for face, count in counts.items() if count == 2]
        return max(pairs) * 2 if pairs else 0

    @staticmethod
    def two_pair(d1, d2, d3, d4, d5):
        counts = Yahtzee._counts(Yahtzee._values(d1, d2, d3, d4, d5))
        pairs = [face for face, count in counts.items() if count == 2]
        return sum(pairs) * 2 if len(pairs) == 2 else 0

    @staticmethod
    def _n_of_a_kind_score(n, dice):
        counts = Yahtzee._counts(dice)
        matches = [face for face, count in counts.items() if count == n]
        return matches[0] * n if matches else 0

    @staticmethod
    def four_of_a_kind(_1, _2, d3, d4, d5):
        return Yahtzee._n_of_a_kind_score(4, Yahtzee._values(_1, _2, d3, d4, d5))

    @staticmethod
    def three_of_a_kind(d1, d2, d3, d4, d5):
        return Yahtzee._n_of_a_kind_score(3, Yahtzee._values(d1, d2, d3, d4, d5))

    @staticmethod
    def smallStraight(d1, d2, d3, d4, d5):
        dice = Yahtzee._values(d1, d2, d3, d4, d5)
        return 15 if set(dice) == {1, 2, 3, 4, 5} else 0

    @staticmethod
    def largeStraight(d1, d2, d3, d4, d5):
        dice = Yahtzee._values(d1, d2, d3, d4, d5)
        return 20 if set(dice) == {2, 3, 4, 5, 6} else 0

    @staticmethod
    def fullHouse(d1, d2, d3, d4, d5):
        dice = Yahtzee._values(d1, d2, d3, d4, d5)
        return sum(dice) if sorted(Yahtzee._counts(dice).values()) == [2, 3] else 0
      