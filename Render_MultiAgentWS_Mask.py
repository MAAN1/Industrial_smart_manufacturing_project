
import plotly.figure_factory as ff

def Render_MultiAgent(n_start_time, n_duration_time, n_bay_start, n_job_id):

	op = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']

	colors = ('rgb(46, 137, 205)',
			  'rgb(114, 44, 121)',
			  'rgb(198, 47, 105)',
			  'rgb(58, 149, 136)',
			  'rgb(107, 127, 135)',
			  'rgb(46, 180, 50)',
			  'rgb(150, 44, 50)',
			  'rgb(10, 34, 58)',
			  'rgb(196, 23, 34)',
			  'rgb(15, 144, 70)',
			  'rgb(180, 94, 20)',
			  'rgb(120, 48, 5)',
			  'rgb(110, 65, 120)',
			  'rgb(77, 32, 89)',
			  'rgb(100, 47, 150)',
			  'rgb(70, 12, 189)',
			  'rgb(177, 132, 29)',
			  'rgb(70, 31, 19)',
			  'rgb(78, 29, 23)',
			  'rgb(37, 132, 89)')

	def create_draw_defination():
		df = []
		for index in range(len(n_job_id)):
			operation = {}
			# Machine, ordinate
			operation['Task'] = 'Workstation' + str(n_bay_start.__getitem__(index)+1)
			operation['Start'] = n_start_time.__getitem__(index)
			operation['Finish'] = n_start_time.__getitem__(index) + n_duration_time.__getitem__(index)


			# Artifact,
			job_num = op.index(n_job_id.__getitem__(index)) + 1
			operation['Resource'] = 'T' + str(job_num)
			#print('operation : ',operation)
			df.append(operation)
		df.sort(key=lambda x: x["Task"], reverse=False)
		return df
	""""
	create_gantt(df, colors=None, index_col=None, show_colorbar=False, reverse_colors=False, title='Gantt Chart', bar_width=0.2, 
	showgrid_x=False, showgrid_y=False, height=600, width=None, tasks=None, task_names=None, data=None, group_tasks=False, show_hover_fill=True)¶	
	"""
	df = create_draw_defination()
	#print("DF", df)
	fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
						  group_tasks=True, showgrid_x = True, showgrid_y = True)
	fig['layout']['xaxis'].update({'type':None})
	fig.layout.font.size = 45
	#fig.add_scatter3d()
	fig.show()
