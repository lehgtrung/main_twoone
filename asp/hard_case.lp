atom(loc("8+9")). % Mexico
atom(loc("10+11")). % Guadalajara
atom(loc("12+13")). % Monterrey
atom(loc("14+15")). % Tijuana
atom(org("16+18")). % Ciudad Juarez
atom(loc("19+20")). % Villahermosa
atom(loc("21+22")). % Hermosillo
atom(loc("23+24")). % Culiacan
atom(loc("25+26")). % Morelia
atom(loc("27+28")). % Veracruz
atom(loc("31+32")). % Torreon
atom(loc("33+34")). % Puebla
atom(loc("35+36")). % Merida
atom(loc("37+38")). % Chihuahua
atom(loc("39+40")). % Oaxaca
atom(loc("41+42")). % Tampico
atom(loc("43+44")). % Leon
atom(loc("45+46")). % Tapachula.
atom(org("46+47")). % Indemerc
atom(org("48+49")). % Diego
atom(org("58+61")). % Indemerc-Louis Harris firm
atom(loc("67+68")). % Valley
atom(peop("74+77")). % Fernandez de Cevallos
atom(orgbasedIn("48+49","31+32")). % Diego(org); Torreon(loc)
atom(locatedIn("19+20","8+9")). % Villahermosa(loc); Mexico(loc)
atom(orgbasedIn("48+49","45+46")). % Diego(org); Tapachula(loc)
atom(orgbasedIn("48+49","10+11")). % Diego(org); Guadalajara(loc)
atom(locatedIn("31+32","33+34")). % Torreon(loc); Puebla(loc)
atom(locatedIn("27+28","31+32")). % Veracruz(loc); Torreon(loc)
atom(liveIn("74+77","10+11")). % Fernandez de Cevallos(peop); Guadalajara(loc)
atom(locatedIn("10+11","8+9")). % Guadalajara(loc); Mexico(loc)
atom(liveIn("16+18","8+9")). % Ciudad Juarez(org); Mexico(loc) => VIOLATED
atom(locatedIn("33+34","35+36")). % Puebla(loc); Merida(loc)
atom(orgbasedIn("48+49","14+15")). % Diego(org); Tijuana(loc)
atom(orgbasedIn("48+49","33+34")). % Diego(org); Puebla(loc)
atom(locatedIn("12+13","8+9")). % Monterrey(loc); Mexico(loc)
atom(liveIn("74+77","8+9")). % Fernandez de Cevallos(peop); Mexico(loc)
atom(locatedIn("25+26","27+28")). % Morelia(loc); Veracruz(loc)
atom(liveIn("74+77","27+28")). % Fernandez de Cevallos(peop); Veracruz(loc)
atom(orgbasedIn("48+49","8+9")). % Diego(org); Mexico(loc)
atom(orgbasedIn("48+49","27+28")). % Diego(org); Veracruz(loc)