CC = nvcc -std=c++11

all: read_hemi.out

read_hemi.out: read_hemi.o shader.o controls.o
	$(CC) -I/usr/include/gifti -I/usr/include/nifti -I/usr/include/GLFW \
		-I/usr/local/cuda/samples/common/inc \
		read_hemi.o shader.o controls.o \
		-lgiftiio -lglut -lGL -lGLU -lGLEW -lglfw \
		-o read_hemi.out

read_hemi.o: read_hemi.cu
	$(CC) -I/usr/include/gifti -I/usr/include/nifti -I/usr/include/GLFW \
	 			-I/usr/local/cuda/samples/common/inc \
				-c read_hemi.cu

shader.o: shader.cc shader.h
	$(CC) -c shader.cc

controls.o: controls.cc controls.h
	$(CC) -I/usr/include/GLFW -c controls.cc

clean:
	rm *.o
	rm read_hemi
