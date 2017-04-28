#include "color_code_me.h"

void print_warning(char* warning_message)
{
	printf("\x1b[1m\x1b[33m%s\x1b[0m\n", warning_message);
}

void print_success(char* success_message)
{
	printf("\x1b[1m\x1b[32m%s\x1b[0m\n", success_message);
}

void print_error(char* error_message)
{
	printf("\x1b[1m\x1b[31m%s\x1b[0m\n", error_message);
}

void print_color(char* string, char* color)
{
	printf("%s\n", string);
}
