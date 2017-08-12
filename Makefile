CC = g++ -std=c++11

read_hemi: read_hemi.o shader.o controls.o
	$(CC) -I/usr/include/gifti -I/usr/include/nifti -I/usr/include/GLFW \
		read_hemi.cc shader.o controls.o \
		-lgiftiio -lglut -lGL -lGLU -lGLEW -lglfw \
		-o read_hemi

read_hemi.o: read_hemi.cc
	$(CC) -I/usr/include/gifti -I/usr/include/nifti -I/usr/include/GLFW -c read_hemi.cc

shader.o: shader.cc shader.h
	$(CC) -c shader.cc

controls.o: controls.cc controls.h
	$(CC) -I/usr/include/GLFW -c controls.cc

clean:
	rm *.o
	rm read_hemi
