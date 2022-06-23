#             cd Documents/ScriptPython
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})
def Render_Res_MultiAgent(n_start_time, n_duration_time, n_bay_start, n_job_id):
	op = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']

	colors = ('rgb(152, 223, 138)',
			  'rgb(227, 119, 194)',
			  'rgb(23, 190, 207)',
			  'rgb(58, 149, 136)',
			  'rgb(107, 127, 135)',
			  'rgb(46, 180, 50)',
			  'rgb(150, 44, 50)',
			  'rgb(100, 47, 150)',
			  'rgb(152, 223, 138)',
			  'rgb(227, 119, 194)',
			  'rgb(23, 190, 207)',
			  'rgb(58, 149, 136)',
			  'rgb(107, 127, 135)',
			  'rgb(46, 180, 50)',
			  'rgb(150, 44, 50)',
			  'rgb(100, 47, 150)')
	#millis_seconds_per_minutes = 1000 * 60
	#start_time = time.time() * 1000
	job_sumary = {}
	def create_draw_defination():
		df = []
		for index in range(len(n_job_id)):
			operation = {}
			# Machine, ordinate
			operation['Task'] = 'Workstation' + str(n_bay_start.__getitem__(index) + 1)
			operation['Start'] =n_start_time.__getitem__(index)
			operation['Finish'] = n_start_time.__getitem__(index) + n_duration_time.__getitem__(index)
			# Artifact,
			job_num = op.index(n_job_id.__getitem__(index)) + 1
			operation['Resource'] = 'R' + str(job_num)
			#print('operation : ',operation)
			df.append(operation)
		df.sort(key=lambda x: x["Task"], reverse=True)
		return df
	
	df = create_draw_defination()
	#print('df : ',df)
	fig = ff.create_gantt(df,colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x = True, showgrid_y = True)
	fig['layout']['xaxis'].update({'type': None})
	fig.layout.font.size = 45
	fig.show()
