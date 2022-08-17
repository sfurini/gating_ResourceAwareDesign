from matplotlib.backends.backend_pdf import PdfPages

import gating

# pdf file that will include all the plots
pdf = PdfPages('figures.pdf')

# just create an instance of the class FcsData
D = gating.FcsData()

# add exp data from example.fcs
# data from more than one file can be added
D.add_sample('./example.fcs')

# apply logical transformation to the data of feature BL1-A and YL2-A
D.set_data(['BL1-A', 'YL2-A'])
D.fit_transform()
D.apply_transform()

# find optimal 1D gates for the two features
D.gate1D(pdf, 'BL1-A', 200) # 200 is the threshold for manual gating
D.gate1D(pdf, 'YL2-A', 200) # 200 is the threshold for manual gating

# apply 2D gating strategies
#   manual_gate = tuple with:
#       1st element the average of BL1-A in the or-gate
#       2nd element the average of YL2-A in the or-gate
#       3rd element the percentage of cell in the or-gate
#   manual_gate_geom = tuple with:
#       1st element the geometric mean of BL1-A in the or-gate
#       2nd element the geometric mean of YL2-A in the or-gate
#       3rd element the percentage of cell in the or-gate
#   manual_gate_meadian = tuple with:
#       1st element the median of BL1-A in the or-gate
#       2nd element the median of YL2-A in the or-gate
#       3rd element the percentage of cell in the or-gate
#   fuzzy = tuple with:
#       1st element the weighted average of BL1-A as computed by the automatic gating algorithm
#       2nd element the weighted average of YL2-A as computed by the automatic gating algorithm
#       3rd element the percentage of cell in the or-gate
manual_gate, manual_gate_geom, manual_gate_median, fuzzy = D.gate2D(pdf, ['BL1-A', 'YL2-A'], [200, 200], what = 'or') # [200, 200] are the thresholds for manual gating for the two features
print('Manual OR gate (mean): BL1-A = {}, YL2-A = {} fraction of cells in the gate = {}'.format(manual_gate[0], manual_gate[1], manual_gate[2]))
print('Manual OR gate (geometric mean): BL1-A = {}, YL2-A = {}'.format(manual_gate_geom[0], manual_gate_geom[1]))
print('Manual OR gate (median): BL1-A = {}, YL2-A = {}'.format(manual_gate_median[0], manual_gate_median[1]))
print('Gating with fuzzy clustering algorithm: BL1-A = {}, YL2-A = {} fraction of cells in the gate = {}'.format(fuzzy[0], fuzzy[1], fuzzy[2]))

pdf.close()
