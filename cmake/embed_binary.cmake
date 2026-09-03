# Embeds a binary file as a C unsigned-char array, replacing xxd -i so the
# Vulkan SPIR-V shaders build on hosts without xxd (Windows, minimal
# containers). Usage:
#   cmake -DINPUT=<file> -DOUTPUT=<header> -DSYMBOL=<name> -P embed_binary.cmake
file(READ "${INPUT}" HEX_CONTENT HEX)
string(LENGTH "${HEX_CONTENT}" HEX_LENGTH)
math(EXPR BYTE_COUNT "${HEX_LENGTH} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," BYTES "${HEX_CONTENT}")
string(REGEX REPLACE "(0x[0-9a-f][0-9a-f],){12}" "\\0\n  " BYTES "${BYTES}")
file(WRITE "${OUTPUT}"
     "unsigned char ${SYMBOL}[] = {\n  ${BYTES}\n};\nunsigned int ${SYMBOL}_len = ${BYTE_COUNT};\n")
