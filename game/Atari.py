from ale_python_interface import ALEInterface
import numpy as np 
import cv2
from random import randrange

class Atari:
	def __init__(self,rom_name):
		self.ale = ALEInterface()
		self.max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode")
		self.ale.setInt("random_seed",123)
		self.ale.setInt("frame_skip",4)
		self.ale.loadROM('game/' +rom_name)
		self.screen_width,self.screen_height = self.ale.getScreenDims()
		self.legal_actions = self.ale.getMinimalActionSet()
		self.action_map = dict()
		for i in range(len(self.legal_actions)):
			self.action_map[self.legal_actions[i]] = i
		#print len(self.legal_actions)
		self.windowname = rom_name
		cv2.startWindowThread()
		cv2.namedWindow(rom_name)

	def get_image(self):
		numpy_surface = np.zeros(self.screen_height*self.screen_width*3, dtype=np.uint8)
		self.ale.getScreenRGB(numpy_surface)
		image = np.reshape(numpy_surface, (self.screen_height, self.screen_width, 3))
		return image

	def newGame(self):
		self.ale.reset_game()
		return self.get_image()

	def next(self, action):
		reward = self.ale.act(self.legal_actions[np.argmax(action)])	
		nextstate = self.get_image()
		
		cv2.imshow(self.windowname,nextstate)
		if self.ale.game_over():
			self.newGame()
		#print "reward %d" % reward 
		return nextstate, reward, self.ale.game_over()
