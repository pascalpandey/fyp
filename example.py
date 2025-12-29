def get_user_input(prompt_text: str = "Enter something: ") -> str:
	"""Prompt the user and return their input as a stripped string.

	Kept intentionally simple so it works in interactive shells and when
	the script is run from the command line.
	"""
	try:
		# Python input() already prints the prompt; strip trailing newlines/spaces
		return input(prompt_text).strip()
	except EOFError:
		# Gracefully handle non-interactive environments
		return ""


def main() -> None:
	"""Minimal CLI demo that uses get_user_input and prints what was entered."""
	user_text = get_user_input("Please type a short message and press Enter: ")
	if user_text:
		print(f"You entered: {user_text}")
	else:
		print("No input received.")


if __name__ == "__main__":
	main()

