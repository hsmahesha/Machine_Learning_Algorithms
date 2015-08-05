################################################################################
#                                                                              #
#                        Classification Tree Printer:                          #
#                                                                              #
################################################################################
#                                                                              #
################################################################################



#------------------------------------------------------------------------------#
# import built-in system modules here                                          #
#------------------------------------------------------------------------------#
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Print classification tree                                                    #
#------------------------------------------------------------------------------#
class PrintTree:
      root = None
      def __init__(self, root=None):
          self.root = root

      def get_width(self, tree):
          if tree.tnode == None and tree.fnode == None:
             return 1
          else:
             tw = 0
             fw = 0
             if tree.tnode != None:
                tw = self.get_width(tree.tnode)
             if tree.fnode != None:
                fw = self.get_width(tree.fnode)
             return tw + fw

      def get_depth(self, tree):
          if tree.tnode == None and tree.fnode == None:
             return 0
          else:
             td = 0
             fd = 0
             if tree.tnode != None:
                td = self.get_depth(tree.tnode)
             if tree.fnode != None:
                fd = self.get_depth(tree.fnode)
             return max(td, fd)

      def draw_image_node(self, draw, tree, x, y):
          if tree.leaf_node == False:
             w1 = self.get_width(tree.fnode) * 100
             w2 = self.get_width(tree.tnode) * 100

             left = x-(w1+w2)/2.0
             right = x+(w1+w2)/2.0

             draw.text((x-20,y-10), str(tree.col)+":"+str(tree.value), (0,0,0))

             draw.line((x,y,left+w1/2.0,y+100), fill=(255,0,0))
             draw.line((x,y,right-w2/2.0,y+100), fill=(255,0,0))

             if tree.fnode != None:
                self.draw_image_node(draw, tree.fnode, left+w1/2.0, y+100)
             if tree.tnode != None:
                self.draw_image_node(draw, tree.tnode, right-w2/2.0, y+100)
          else:
             txt = ' \n'.join(['%s:%d'%v for v in tree.results.items()])
             draw.text((x-20,y), txt, (0,0,0))

      def draw_image_tree(self, tree, jpeg='./data_set/class_tree/class_tree.jpg'):
          if tree == None:
             return

          w = self.get_width(tree) * 1000
          h = self.get_depth(tree) * 1000 + 1200

          img = Image.new('RGB', (w,h), (255, 255, 255))
          draw = ImageDraw.Draw(img)

          self.draw_image_node(draw, tree, w/2.0, 20)
          img.save(jpeg, 'JPEG')

      def draw_tree(self):
          self.draw_image_tree(self.root)

      def print_text_tree(self, tree, indent=''):
          if tree.leaf_node == True:
             print(indent+str(tree.results))
          else:
             print(indent+str(tree.col)+":"+str(tree.value)+"?")
             if tree.tnode != None:
                print(indent+"T->")
                self.print_text_tree(tree.tnode, indent+' ')
             if tree.fnode != None:
                print(indent+"F->")
                self.print_text_tree(tree.fnode, indent+' ')

      def print_tree(self):
          self.print_text_tree(self.root)
#------------------------------------------------------------------------------#
