"""AssignPronType block."""
from udapi.core.block import Block
from collections import Counter


class AssignPronType(Block):

    """ This block collects all lemmas (or word forms, if the treebank
    lacks lemmas) of pronouns and determiners in the treebank and examines
    each pronoun/determiner/adverb node, checking whether it lacks the
    “PronType” feature. If it does, then the block decides, based on the
    lemma (or word form) what value of “PronType” the node should have,
    and adds the value to the features of the node.
    """

    def process_node(self, node):
        # 'lui-même' and its variants may also be emphatic determiners based on the nature of their parent nodes
        Prs = ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
                'me', 'se', 'te', 'le', 'les', 'nous', 'vous', 'leur', 'leurs',
                'y', 'lui', 'en', 'moi', 'toi', 'lui', 'eux',
                'elle-même', 'eux-mêmes', 'lui-même', 'moi-même', 'soi-même']
        Poss = ['mien', 'mienne', 'miens', 'miennes', 'tien', 'tienne',
                'tiens', 'tiennes', 'sien', 'sienne', 'nôtre', 'vôtre',
                'nôtres', 'vôtres', 'leur', 'leurs', 'mon', 'ma', 'soi']
        # In the French GSD treebank, emphatic determiners are tagged as PRON (so need to deal with them)
        Emp = ['elle-même', 'eux-mêmes', 'lui-même', 'moi-même', 'soi-même']
        Dem = ['ce', "c'", 'ça', 'cela', 'ceci', 'celui', 'celle', 'celles',
                'celui-là', 'celui-ci', 'celle-là', 'ceux', 'ceux-ci', 'ceux-là',
                'celles-ci', 'celles-là', 'dernier']
        Int_Rel = ['qui', 'que', 'quoi', 'quand', 'où', 'comment', 'pourquoi',
                'lequel', 'duquel', 'auquel', 'lesquels', 'desquels', 'auxquels',
                'laquelle', 'lesquelles', 'desquelles', 'auxquelles', 'quelle',
                'quel', 'quelles', 'quels', 'quoi', 'combien']
        # 'quel' and its variants can both be Int or Exc and it is not readily apparent how to choose one automatically
        Exc = ['quel', 'que']
        # Omitting two partitive articles 'de la' and 'de l''
        Art = ['le', 'Le','la', 'La', 'les', 'Les', 'un', 'une', 'des', 'du']
        # 'on' [fr] can either mean 'we' [eng] (personal pronoun) or 'one' (indefinite) [eng] as in 'one can do [...] '
        # This is handled in the conditional statements below
        Ind = ['autre', 'autrui', 'certain', 'chacun', 'même', 'on',
                'plusieurs', 'plupart', 'beaucoup', 'dernier', "quelqu'un",
                'un', 'tel', 'différent', 'divers', 'quelque']
        Neg = ['rien', 'nul', 'aucun']
        Tot = ['tout', 'chaque']

        prontype = node.feats['PronType']
        # Many DET tags are inconsistent in this treebank
        if (node.upos == 'PRON' or node.upos == 'DET') and not prontype:
            if node.lemma in Ind:
                if node.lemma == 'on' and node.parent.upos == 'DET':
                    node.feats['PronType'] = 'Ind'
                else:
                    node.feats['PronType'] = 'Prs'
            elif node.lemma in Emp and node.parent.upos == 'VERB':
                node.feats['PronType'] = 'Emp'
            elif node.lemma in Prs or node.lemma in Poss:
                node.feats['PronType'] = 'Prs'
            elif node.lemma in Dem:
                node.feats['PronType'] = 'Dem'
            elif node.lemma in Neg:
                node.feats['PronType'] = 'Neg'
            elif node.lemma in Tot:
                node.feats['PronType'] = 'Tot'
