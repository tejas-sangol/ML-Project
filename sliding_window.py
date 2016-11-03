def sliding_window(width=320,height=240,offset=30,size=128):
	for i in range(0,width-size,offset):
		for j in range(0,height-size,offset):
			yield ( i,j,i+size,j+size)
