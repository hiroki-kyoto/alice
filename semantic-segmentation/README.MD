<font face='Centaur' size=4 color=#c00>

### Semantic Segmentation

---

<font color=#c00>

Guided Learning is described in my [blog](https://www.cnblogs.com/thisisajoke/p/11336448.html)

Mainly, I am using this new technique which is invented by my own,
to explore the limit of semantic segmentation.

The to-do list is:

1. Select a single object, and place it with various poses under a 
clean background, such as a white wall. Pose number requires at
least 100;
2. Collecting photos shot for different backgrounds, anythings 
could be fine to be contained in such backgrounds, and make the
background as complex as possible, prepare at least 100;
3. Train separately a model for part-missing conditioned auto
encoder, to get the function of co-activation of neurons which
respond to theirs related patterns of this very object.
4. Train separately a model for coherent representation of this
specific object.
5. Train separately a model for detecting the object with single
pose of the object placed over every backgrounds.
6. Combine the 3 networks together to enable a free-background 
detecting and semantic segmentation functionality. 

</font>