import logging

from lingua import Language, LanguageDetectorBuilder


# https://github.com/pemistahl/lingua-py
class LanguageDetector:

    def __init__(self, languages=None):
        if languages is None:
            logging.debug("No languages provided, using default languages ITALIAN and ENGLISH")
            languages = [Language.ITALIAN, Language.ENGLISH]
        self.languages = languages

    def detect_language_of(self, text):
        detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        language = detector.detect_language_of(text)
        return language.iso_code_639_1.name

        
if __name__ == "__main__":

    detector = LanguageDetector([Language.ITALIAN, Language.ENGLISH])
    text = "Ciao, come stai?"
    print(detector.detect_language_of(text))