import numpy as np

score = {}
global secret_dict

def similarities(word1, word2):
	if len(word1)!=len(word2):
		return 0
	
	sim = 0
	for i,x in enumerate(word1):
		if word1[i] == word2[i]:
			sim = sim+1
	
	return sim

def create_score():

	for i,x in enumerate(secret_dict):
		score[secret_dict[i]] = {}
		for j in range(i,len(secret_dict)):
			score[secret_dict[i]][secret_dict[j]] = similarities(secret_dict[i], secret_dict[j])

	return

def give_query_idx(redu_word_idx):

	my_score = []
	for i in redu_word_idx:
		scorei = 0
		for j in redu_word_idx:
			if i<j:
				scorei += score[secret_dict[i]][secret_dict[j]]
			else:
				scorei += score[secret_dict[j]][secret_dict[i]]
		my_score.append(scorei)

	# return redu_word_idx[my_score.index(max(my_score))]

	k = max(my_score)
	max_list = []

	for i,x in enumerate(my_score):
		if my_score[i] == k:
			max_list.append(i)

	idx = np.random.randint( 0, len(max_list))

	return redu_word_idx[max_list[idx]]

def assign_secret_dict(words):
	global secret_dict 
	secret_dict = words

	return 
	
def my_fit( words, verbose = False ):
	assign_secret_dict(words)
	create_score()
	dt = Tree( min_leaf_size = 1, max_depth = 15 )
	dt.fit( verbose )
	return dt


class Tree:
	def __init__( self, min_leaf_size, max_depth ):
		self.root = None
		self.words = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit( self, verbose = False ):
		self.words = secret_dict
		self.root = Node( depth = 0, parent = None )
		if verbose:
			print( "root" )
			print( "└───", end = '' )
		# The root is trained with all the words
		self.root.fit( my_words_idx = np.arange( len( self.words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth, verbose = verbose )


class Node:
	# A node stores its own depth (root = depth 0), a link to its parent
	# A link to all the words as well as the words that reached that node
	# A dictionary is used to store the children of a non-leaf node.
	# Each child is paired with the response that selects that child.
	# Note: my_words_idx only stores indices and not the words themselves
	def __init__( self, depth, parent ):
		self.depth = depth
		self.parent = parent
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query_idx = None
	
	# Each node must implement a get_query method that generates the
	# query that gets asked when we reach that node. Note that leaf nodes
	# also generate a query which is usually the final answer
	def get_query( self ):
		return self.query_idx
	
	# Each non-leaf node must implement a get_child method that takes a
	# response and selects one of the children based on that response
	def get_child( self, response ):
		# This case should not arise if things are working properly
		# Cannot return a child if I am a leaf so return myself as a default action
		if self.is_leaf:
			print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
			child = self
		else:
			# This should ideally not happen. The node should ensure that all possibilities
			# are covered, e.g. by having a catch-all response. Fix the model if this happens
			# For now, hack things by modifying the response to one that exists in the dictionary
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child
	
	# Dummy leaf action -- just return the first word
	def process_leaf( self, my_words_idx):
		return my_words_idx[0]
	
	def reveal( self, word, query ):
		# Find out the intersections between the query and the word
		mask = [ *( '_' * len( word ) ) ]
		
		for i in range( min( len( word ), len( query ) ) ):
			if word[i] == query[i]:
				mask[i] = word[i]
		
		return ' '.join( mask )
	
	# Dummy node splitting action -- use a random word as query
	# Note that any word in the dictionary can be the query
	def process_node( self, my_words_idx, verbose ):
		# For the root we do not ask any query -- Melbot simply gives us the length of the secret word
		if self.parent == None:
			query_idx = -1
			query = ""
		else:
			query_idx = give_query_idx(my_words_idx)
			# query_idx = my_words_idx[np.random.randint( 0, len( my_words_idx ) )]
			query = secret_dict[ query_idx ]
		
		split_dict = {}
		
		for idx in my_words_idx:
			mask = self.reveal( secret_dict[ idx ], query )
			if mask not in split_dict:
				split_dict[ mask ] = []
			
			split_dict[ mask ].append( idx )
		
		if len( split_dict.items() ) < 2 and verbose:
			print( "Warning: did not make any meaningful split with this query!" )
		
		return ( query_idx, split_dict )
	
	def fit( self, my_words_idx, min_leaf_size, max_depth, fmt_str = "    ", verbose = False ):
		self.my_words_idx = my_words_idx
		
		# If the node is too small or too deep, make it a leaf
		# In general, can also include purity considerations into account
		if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth:
			self.is_leaf = True
			self.query_idx = self.process_leaf( self.my_words_idx)
			if verbose:
				print( '█' )
		else:
			self.is_leaf = False
			( self.query_idx, split_dict ) = self.process_node( self.my_words_idx, verbose )
			
			if verbose:
				print( secret_dict[ self.query_idx ] )
			
			for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
				if verbose:
					if i == len( split_dict ) - 1:
						print( fmt_str + "└───", end = '' )
						fmt_str += "    "
					else:
						print( fmt_str + "├───", end = '' )
						fmt_str += "│   "
				
				# Create a new child for every split
				self.children[ response ] = Node( depth = self.depth + 1, parent = self )
				
				# Recursively train this child node
				self.children[ response ].fit( split, min_leaf_size, max_depth, fmt_str, verbose )