import argparse
import plotly.graph_objects as go


def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]


def create_dotplot(seq1, seq2, dot_size):

    # remove whitespace
    seq1 = seq1.replace(" ", "")
    seq2 = seq2.replace(" ", "")

    # convert the string to char list
    seq1_list = list(seq1)
    seq2_list = list(seq2)

    # find all the occurences of the chars in seq1, in seq2
    matches = [findOccurrences(seq2, c) for c in seq1]

    # create lists of all the matches found, to be used in plotting
    # (need to do this to keep account of multiple matches)
    match_lists = list()
    for i in range(len(max(matches, key=len))):
        c_list = list()
        for l in matches:
            if len(l) > i:
                c_list.append(l[i])
            else:
                c_list.append(None)
        match_lists.append(c_list)

    fig = go.Figure()

    # plot all the match lists
    for l in match_lists:
        fig.add_trace(go.Scatter(
            x=l,
            y=list(range(len(seq1_list))),
            name='Match',
            marker=dict(
                color='rgba(156, 165, 196, 0.95)',
                line_color='rgba(156, 165, 196, 1.0)',
            )
        ))

    fig.update_traces(mode='markers', marker=dict(
        line_width=1, symbol='circle', size=dot_size))

    fig.update_layout(
        title="Sequence Similarity",
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgb(102, 102, 102)',
            tickfont_color='rgb(102, 102, 102)',
            showticklabels=True,
            ticks='outside',
            tickcolor='rgb(102, 102, 102)',
            tickmode='array',
            tickvals=list(range(len(seq2_list))),
            ticktext=seq2_list
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(seq1_list))),
            ticktext=seq1_list
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest',
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showspikes=True, side="top")
    fig.update_yaxes(showspikes=True, autorange="reversed")
    fig.show()


seq1 = "This is a short text No Copy"
seq2 = "This is a long text Copy Not Please"

# seq1 = """MALDGIRMPDGCYADGTWELSVHVTDLNRDVTLRVTGEVHIGGVMLKLVEKLDVKKDWSD
# HALWWEKKRTWLLKTHWTLDKYGIQADAKLQFTPQHKLLRLQLPNMKYVKVKVNFSDRVF
# KAVSDICKTFNIRHPEELSLLKKPRDPTKKKKKKLDDQSEDEALELEGPLITPGSGSIYS
# SPGLYSKTMTPTYDAHDGSPLSPTSAWFGDSALSEGNPGILAVSQPITSPEILAKMFKPQ
# ALLDKAKINQGWLDSSRSLMEQDVKENEALLLRFKYYSFFDLNPKYDAIRINQLYEQAKW
# AILLEEIECTEEEMMMFAALQYHINKLSIMTSENHLNNSDKEVDEVDAALSDLEITLEGG
# KTSTILGDITSIPELADYIKVFKPKKLTLKGYKQYWCTFKDTSISCYKSKEESSGTPAHQ
# MNLRGCEVTPDVNISGQKFNIKLLIPVAEGMNEIWLRCDNEKQYAHWMAACRLASKGKTM
# ADSSYNLEVQNILSFLKMQHLNPDPQLIPEQITTDITPECLVSPRYLKKYKNKQPGYIRD
# LITARILEAHQNVAQMSLIEAKMRFIQAWQSLPEFGITHFIARFQGGKKEELIGIAYNRL
# IRMDASTGDAIKTWRFSNMKQWNVNWEIKMVTVEFADEVRLSFICTEVDCKVVHEFIGGY
# IFLSTRAKDQNESLDEEMFYKLTSGWV"""

# seq2 = """MAGMKTASGDYIDSSWELRVFVGEEDPEAESVTLRVTGESHIGGVLLKIVEQINRKQDWS
# DHAIWWEQKRQWLLQTHWTLDKYGILADARLFFGPQHRPVILRLPNRRALRLRASFSQPL
# FQAVAAICRLLSIRHPEELSLLRAPEKKEKKKKEKEPEEELYDLSKVVLAGGVAPALFRG
# MPAHFSDSAQTEACYHMLSRPQPPPDPLLLQRLPRPSSLSDKTQLHSRWLDSSRCLMQQG
# IKAGDALWLRFKYYSFFDLDPKTDPVRLTQLYEQARWDLLLEEIDCTEEEMMVFAALQYH
# INKLSQSGEVGEPAGTDPGLDDLDVALSNLEVKLEGSAPTDVLDSLTTIPELKDHLRIFR
# PRKLTLKGYRQHWVVFKETTLSYYKSQDEAPGDPIQQLNLKGCEVVPDVNVSGQKFCIKL
# LVPSPEGMSEIYLRCQDEQQYARWMAGCRLASKGRTMADSSYTSEVQAILAFLSLQRTGS
# GGPGNHPHGPDASAEGLNPYGLVAPRFQRKFKAKQLTPRILEAHQNVAQLSLAEAQLRFI
# QAWQSLPDFGISYVMVRFKGSRKDEILGIANNRLIRIDLAVGDVVKTWRFSNMRQWNVNW
# DIRQVAIEFDEHINVAFSCVSASCRIVHEYIGGYIFLSTRERARGEELDEDLFLQLTGGH
# EAF"""

# seq1 = """CCACCTGTCTGGTCAGGCTTGCTCCAGAACTCTGCATAGATAGACAGTTGCAGACAGGAAGGGATTCTCCCTGGCCCCGGGTATCAGTGACGAATTTACCTAAGAGGCAGAGAAAAAGGC"""

# seq2 = """ACTCGGAGCGAGAGCAAGACCTAGCAGGCTGGGCTGGACGTGCCCCCACCTGCCCGCCGCGCTAGCCTGCGCCGCCAAGCTGTGGCCCCGGGACTCGGACCGTGCTTGGGTGCTGGGAGG"""

seq1 = """To be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune"""
seq2 = """In one of the Bards best thought of tragedies our insistent hero Hamlet queries on two fronts about how life turns rotten"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", help="input your sequences", action="store_true")
parser.add_argument("-s1", "--first", help="first sequence")
parser.add_argument("-s2", "--second", help="second sequence")
parser.add_argument("-d", "--dot_size", type=int, help="dot size", default=8)
args = parser.parse_args()
if args.input:
    if not args.first or not args.second:
        print("Provide both sequences. Default being used.")
    else:
        seq1 = args.first
        seq2 = args.second

create_dotplot(seq1, seq2, dot_size=args.dot_size)
