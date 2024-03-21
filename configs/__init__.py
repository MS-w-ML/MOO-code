import os
import time
import json
import collections
from easydict import EasyDict
if __name__=='__main__':
    import sys
    sys.path[0] = 'e:\\projects\\MOO'
from utils.utils import mkdir, mkdirs, recursive_mkdir

# Set up color boundaries
# "Infinity" is valid
DEFAULT_PARAMS =   {"color_cls_boundary":{
    "red":{"min":595, "max":630}, # 
    "orange":{"min":550, "max":595},
    "yellow":{"min":520, "max":550}, 
    "green":{"min":490, "max":520},
    "cyan":{"min":460, "max":490}, 
    "blue":{"min":420, "max":460},
    "purple":{"min":380, "max":420} # 
    
  }
}

class ConfigManager(object):
	"""Read or write, parse, print configurations.
    """
	def __init__(self, config_dir=None,  verbose=True, mode='synthetic',save2file=True):
		super(ConfigManager, self).__init__()

		if config_dir is not None:
			self.check_validity_config_dir(config_dir)
			self.name = config_dir
			self.opts = self.json_file_to_dict(config_dir)
			self.opts.mode =  mode

			# parse opts
			self.parse_opts()
			self.print_options(verbose)
			if save2file:
				self.pyobj_to_json_file()



	def check_validity_config_dir(self, config_dir):
		assert os.path.exists(config_dir), 'Path to configuration file does not exists: %s'%config_dir
		file_type = config_dir.split('.')[-1]
		assert file_type == 'json', 'Invalid file type: %s. JSON file expected.'%file_type
		
	
	def create_config_file(self, configs, config_dir):

		self.check_validity_config_dir(config_dir)
		temp_dir = "temp.json"
		print("storing configs to file: %s"%(config_dir))

		with open(temp_dir, 'w') as json_file:  
			json.dump(configs, json_file, separators=(',',":"), indent=2)

		self._json_beautify(temp_dir, config_dir)
		print("  Config file created.")


	def json_file_to_pyobj(self,filename):
		def _json_object_hook(d): return collections.namedtuple('ConfigX', d.keys())(*d.values())
		def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

		return json2obj(open(filename).read())

	def json_file_to_dict(self,filename):
		data_dict = json.loads((open(filename).read()))
		return EasyDict(data_dict)

	def pyobj_to_json_file(self, to_dir=None, filename=None):

		if to_dir is None and hasattr(self.opts, 'experiment_dir') and hasattr(self.opts, 'experiment_name'):
			to_dir = self.opts.experiment_dir
		else:
			to_dir = '.'
		if filename is None:
			filename = 'config.json'

		file_path = os.path.join(to_dir, filename)
		print('saving configuration file to %s'%file_path)
		#dict_to_save = self._pyobj_to_dict(self.opts)

		with open(file_path , 'w') as f:
			json.dump(dict(self.opts), f, indent=2, sort_keys=False)
		
		return file_path
		

	def parse_opts(self):
		"""Parse opts, create checkpoints directories"""

		# create base directory
		overwrite = not self.opts.continue_train
		ck_dir = self.opts.checkpoints_dir
		if (not os.path.exists(ck_dir)) or overwrite:
			recursive_mkdir(ck_dir)

		exp_dir = os.path.join(ck_dir, self.opts.experiment_name)
		ds_opt = self.opts.dataset
		
		if self.opts.mode == 'synthetic':
			if self.opts.dataset.type == 'MOO':
				folder_name = 'color%s%s'%(ds_opt.simulator_moo.f_color.type[:2].upper(), ds_opt.simulator_moo.f_color.name[:3].lower()) + \
								'_yield%s%s'%(ds_opt.simulator_moo.f_yield.type[:2].upper(), ds_opt.simulator_moo.f_yield.name[:3].lower())

			elif self.opts.dataset.type =='SO':
				folder_name = '%s_%s_%s%s'%(ds_opt.name, ds_opt.y_type, ds_opt.simulator_so.type[:3].upper(), ds_opt.simulator_so.name[:4].lower())
			else:
				raise Exception('Invalid dataset type %s. Should be \'SO\' or \'MOO\''%self.opts.dataset.type)
		elif self.opts.mode == 'real':
			folder_name = 'real'
		else:
			raise Exception('Invalid mode: %s. Shall be \'real\'|\'synthetic\''%self.opts.mode)

		ds_dir = os.path.join(exp_dir, self.opts.dataset.type)
		ds_dir = os.path.join(ds_dir, folder_name)
		recursive_mkdir(ds_dir)
		
		postfix = str(self.opts.postfix) if hasattr(self.opts, 'postfix') else None
		if postfix is None:
			end_dir = os.path.join(ds_dir, self.opts.model.name)
		else:
			end_dir = os.path.join(ds_dir, self.opts.model.name, postfix)

		if (not os.path.exists(end_dir)) or overwrite:
			mkdir(end_dir, overwrite)
		self.opts.experiment_dir = end_dir

		# insert default params
		for key in DEFAULT_PARAMS.keys():
			if not hasattr(self.opts, key):
				setattr(self.opts, key, DEFAULT_PARAMS[key])
		
		# parse color_cls_boundary for numberize possible 'Infinity'
		color_cls_boundary_dict = self.opts.color_cls_boundary

		for color_key in color_cls_boundary_dict.keys():
			# handle min boundary
			min_boundary = color_cls_boundary_dict[color_key]['min']
			if isinstance(min_boundary, str):
				if min_boundary == "-Infinity":
					self.opts.color_cls_boundary[color_key]['min']  = float('-inf')
				else:
					raise Exception('Invalid string input for [color_cls_boundary.%s.min], should be numbers or \'-Infinity\''%min_boundary)

			# handle max boundary
			max_boundary = color_cls_boundary_dict[color_key]['max']
			if isinstance(max_boundary, str):
				if max_boundary == "Infinity":
					self.opts.color_cls_boundary[color_key]['max']  = float('inf')
				else:
					raise Exception('Invalid string input for [color_cls_boundary.%s.max], should be numbers or \'Infinity\''%min_boundary)

		


	def _list2str(self, list, delimiter=','):
		'''Convert list to string.
		E.g. [1,2,3] -> '1,2,3'
		'''
		# Converting integer list to string list 
		str_list = [str(i) for i in list] 
      
		# Join list items using join() 
		return delimiter.join(str_list)

	def _pyobj_to_dict(self, config_data):

		ret = {}
		for field in config_data._fields:
			field_val = getattr(config_data, field)
			if 'ConfigX' in str(type(field_val)):
				ret[field] = self._pyobj_to_dict(field_val)
			else:
				ret[field] = field_val
				# msg += '\t'*recur_lvl + '-%s'%name.split('.')[-1]+'\n'
				# recur_lvl += 1

				# for field_name in config_data._fields:
				# 	temp = getattr(config_data,field_name)
				# 	if 'ConfigX' in str(type(temp)) : msg = _recursive_print(msg, temp, name+'.'+field_name,recur_lvl)
				# 	else: 
				# 		msg += '\t'*recur_lvl+ "-%s: %s "%(field_name, str(temp))+'\n'
		return ret

	def print_options(self, verbose=True):
		

		now = time.strftime("%c")
		msg = ''
		msg += '\nPrinting configurations: %s \n'%now
		msg += '\t------------start------------\n'

		def _recursive_print_pyobj(msg, config_data, name, recur_lvl=1):

			msg += '\t'*recur_lvl + '-%s'%name.split('.')[-1]+'\n'
			recur_lvl += 1

			for field_name in config_data._fields:
				temp = getattr(config_data,field_name)
				if 'ConfigX' in str(type(temp)) : msg = _recursive_print_pyobj(msg, temp, name+'.'+field_name,recur_lvl)
				else: 
					msg += '\t'*recur_lvl+ "-%s: %s "%(field_name, str(temp))+'\n'
			return msg

		def _recursive_print_dict(msg, config_data, name, recur_lvl=1):

			msg += '\t'*recur_lvl + '-%s'%name.split('.')[-1]+'\n'
			recur_lvl += 1

			for field_name in config_data.keys():
				temp = config_data[field_name]
				if 'dict' in str(type(temp)) : 
					msg = _recursive_print_dict(msg, temp, name+'.'+field_name,recur_lvl)
				else: 
					msg += '\t'*recur_lvl+ "-%s: %s "%(field_name, str(temp))+'\n'
			return msg
		if 'dict' in str(type(self.opts)):
			msg = _recursive_print_dict(msg, self.opts, self.name)
		elif 'ConfigX' in str(type(self.opts)):
			msg = _recursive_print_pyobj(msg, self.opts, self.name)


		msg += '\n\t------------end------------'

		# print
		if verbose:
			print(msg)

		# save to disk
		# exp_dir = os.path.join(self.opts.checkpoints_dir, self.opts.experiment_name)
		# file_name = os.path.join(exp_dir, '_opts.txt')
		# with open(file_name, 'wt') as opts_file:
		# 	opts_file.write(msg)
		# 	opts_file.write('\n')


	def get_options(self):
		return EasyDict(self.opts)

	def set_options(self, opts):
		self.opts = opts

	def _json_beautify(self,temp_dir, config_dir):
		'''Format list of lists to be more readable. 
		'''

		bracket = 0
		
		with open(config_dir, 'w') as json_file: 
			for x in open(temp_dir, 'r'):
				temp = x.strip("\n").strip("\t")
				prev_bracket = bracket
				if "]" in temp: bracket-=1 
				if "[" in temp: bracket+=1
				if bracket ==0 and not prev_bracket==1: print(temp); json_file.write(temp+"\n")
				elif bracket ==0 and prev_bracket==1: print(temp.strip(" ")); json_file.write(temp.strip(" ")+"\n")
				elif bracket==1 and prev_bracket==0: print(temp,end=''); json_file.write(temp)
				else: print(temp.strip(" "),end='') ; json_file.write(temp.strip(" "))
		os.remove(temp_dir)

	

if __name__=='__main__':
	# print("\n========= Creating json data =========")
	# CONFIG_FILE = {
	# 	'working_dir':"D:/Desktop@D/meim2venv/meim3",
	# 	'data_dir':{
	# 		'ISLES2017':"D:/Desktop@D/meim2venv/meim3/data/isles2017"
	# 	},
	# 	'some_numbers': [[ 1, 2, 3, 4],[ 1, 2]],
	# 	'some_numbers2': [ 1.2, 3, 'ggwp'],
	# 	'ggw3': [[2,3],[4,4,4],['a','b','c']]
    # }
	# config_dir = './configs/example_config.json'
	# cm = ConfigManager()
	# cm.create_config_file(CONFIG_FILE, config_dir)


	print("\n========= Loading json data =========")
	cm = ConfigManager('./configs/config_pam.json')
	#cm.pyobj_to_json_file()
	
