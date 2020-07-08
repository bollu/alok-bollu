find kernel/ -iname '*.h' -o -iname '*.c' -exec 'cat' '{}' ';' > concat-kernel.c
find block/ -iname '*.h' -o -iname '*.c' -exec 'cat' '{}' ';' >> concat-kernel.c
find certs/ -iname '*.h' -o -iname '*.c' -exec 'cat' '{}' ';' >> concat-kernel.c
find crypto/ -iname '*.h' -o -iname '*.c' -exec 'cat' '{}' ';' >> concat-kernel.c
find init/ -iname '*.h' -o -iname '*.c' -exec 'cat' '{}' ';' >> concat-kernel.c
find ipc/ -iname '*.h' -o -iname '*.c' -exec 'cat' '{}' ';' >> concat-kernel.c
find lib/ -iname '*.h' -o -iname '*.c' -exec 'cat' '{}' ';' >> concat-kernel.c
find virt/ -iname '*.h' -o -iname '*.c' -exec 'cat' '{}' ';' >> concat-kernel.c
