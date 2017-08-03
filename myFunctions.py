def classify_WkTOTAL(row):
	'''
	Classifies the Wk_TOTAL metric as above or below a cutoff of 32.
	'''
	tot = row['Wk_TOTAL']
	if tot >= 32:
		return 1
	else:
		return 0

def final_score(row):
	'''
	Takes in a row, sums up the 10 weekly scores and does a 30-70 weighting of weekly quiz and final exam to calculate the final mark as a score out of 100.
	'''
	week_tot = 0
	for i in range(2,12):
		week_tot += row['Wk{0}'.format(i)]
	score = week_tot / 40. * 30 + row['Exam_TOTAL'] / 40. * 70
	return score

def convert_to_GPA(row):
	'''
	Takes in a score out of 100 (stored in a row as 'final_score' and returns a GPA letter(s) string.
	'''
	final_score = row['final_score']
	if final_score < 50:
		return 'F'
	elif final_score < 60:
		return 'P'
	elif final_score < 70:
		return 'C'
	elif final_score < 80:
		return 'D'
	else:
		return 'HD'

def variation(row, wkA, wkB):
	'''
	Calculates difference in score between wkB and wkA (wkB always later than wkA) and sets to 1 if constant or increased, and 0 if decreased.
	'''
	diff = row[wkB] - row[wkA]
	if diff >= 0:
		return 1
	else:
		return 0

def convert_exam_to_pass_fail(row):
	'''
	Takes in an exam mark out of 40 (stored in a row as 'Exam_TOTAL' and returns a pass or fail string.
	'''
	exam_score = row['Exam_TOTAL']
	if exam_score < 20:
		return 0 #F
	else:
		return 1 #P
