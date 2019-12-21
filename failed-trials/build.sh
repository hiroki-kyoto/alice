gcc -o test.o transform_3d.c -lm
gcc -shared -fPIC -o libtools.so tools.c -lm
