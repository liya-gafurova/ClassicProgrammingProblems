# # # f = "/home/lia/PycharmProjects/grammar-checker/reviews/all_countries.txt"
# # # out = "/home/lia/PycharmProjects/grammar-checker/reviews/countries.txt"
# # # the = "/home/lia/PycharmProjects/grammar-checker/reviews/THE_countries.txt"
# # # the2 = "/home/lia/PycharmProjects/grammar-checker/reviews/THE_countries2.txt"
# # #
# # # with open(out, 'w') as out_file:
# # #     with open("/home/lia/PycharmProjects/grammar-checker/reviews/_countries.txt") as file:
# # #         for line in file.readlines():
# # #             t = line.split('\t')
# # #             print(t[0])
# # #             out_file.write(f"{t[0].lower()}")
# # #
# # # # with open(the) as the_f:
# # # #     with open(the2,  'w') as  to:
# # # #         for line in the_f.readlines():
# # # #             to.write(f"the {line.lower()}")
#
import json

with open("/home/lia/PycharmProjects/grammar-checker/reviews/_countries.txt") as f:
    l = f.read().splitlines()
    print(json.dumps({'_countries': l}, ensure_ascii=False, ))
    t = json.dumps({'_countries': l}, ensure_ascii=False, )

with open("/home/lia/PycharmProjects/grammar-checker/reviews/the_countries.txt") as f:
    l = f.read().splitlines()
    print(json.dumps({'the_countries': l}, ensure_ascii=False, ))

import json


_COUNTRIES = json.loads("""{"_countries": ["afghanistan", "albania", "algeria", "american samoa", "andorra", "angola", "anguilla", "antarctica", "antigua and barbuda", "argentina", "armenia", "aruba", "australia", "austria", "azerbaijan", "bahrain", "bangladesh", "barbados", "belarus", "belgium", "belize", "benin", "bermuda", "bhutan", "bolivia", "bosnia and herzegovina", "botswana", "brazil", "brunei darussalam", "bulgaria", "burkina faso", "burundi", "cambodia", "cameroon", "canada", "cape verde", "chad", "chile", "china", "christmas island", "cocos islands", "colombia", "comoros", "congo", "costa rica", "côte d'ivoire", "croatia", "cuba", "cyprus", "denmark", "djibouti", "dominica", "east timor", "ecuador", "egypt", "el salvador", "equatorial guinea", "eritrea", "estonia", "ethiopia", "fiji", "finland", "france", "french guiana", "french polynesia", "french southern territories", "gabon", "the gambia", "georgia", "germany", "ghana", "gibraltar", "greece", "greenland", "grenada", "guadeloupe", "guam", "guatemala", "guinea", "guinea-bissau", "guyana", "haiti", "holy see", "honduras", "hong kong", "hungary", "iceland", "india", "indonesia", "iran", "iraq", "ireland", "israel", "italy", "ivory coast", "jamaica", "japan", "jordan", "kazakhstan", "kenya", "kiribati", "korea", "kosovo flag", "kosovo", "kuwait", "kyrgyzstan", "lao", "latvia", "lebanon", "lesotho", "liberia", "libya", "liechtenstein", "lithuania", "luxembourg", "macau", "madagascar", "malawi", "malaysia", "mali", "malta", "martinique", "mauritania", "mauritius", "mayotte", "mexico", "micronesia, federal states of", "moldova, republic of", "monaco", "mongolia", "montenegro", "montserrat", "morocco", "mozambique", "myanmar, burma", "namibia", "nauru", "nepal", "netherlands antilles", "new caledonia", "new zealand", "nicaragua", "niger", "nigeria", "niue", "north macedonia", "norway", "oman", "pakistan", "palau", "palestinian territories", "panama", "papua new guinea", "paraguay", "peru", "pitcairn island", "poland", "portugal", "puerto rico", "qatar", "reunion island", "romania", "russia", "rwanda", "saint kitts and nevis", "saint lucia", "saint vincent and the grenadines", "samoa", "san marino", "sao tome and principe", "saudi arabia", "senegal", "serbia", "seychelles", "sierra leone", "singapore", "slovakia", "slovenia", "somalia", "south africa", "south sudan", "spain", "sri lanka", "sudan", "suriname", "swaziland", "sweden", "switzerland", "syria", "taiwan", "tajikistan", "tanzania", "thailand", "tibet", "timor-leste", "togo", "tokelau", "tonga", "trinidad and tobago", "tunisia", "turkey", "turkmenistan", "tuvalu", "uganda", "uruguay", "uzbekistan", "vanuatu", "vatican city state", "venezuela", "vietnam", "wallis and futuna islands", "western sahara", "yemen", "zambia", "zimbabwe"]}""")['_countries']
THE_COUNTRIES = json.loads("""{"the_countries": ["the bahamas", "the cayman islands", "the central african republic", "the channel islands", "the czech republic", "the cook islands", "the falkland islands", "the faroe islands", "the dominican republic", "the falkland islands", "the gambia", "the isle of man", "the ivory coast", "the leeward islands", "the maldives ", "the maldive islands", "the marshall islands", "the netherlands", "the netherlands antilles", "the philippines", "the solomon islands", "the turks and caicos islands", "the united arab emirates", "the united kingdom ", "the united states", "the united states of america", "the ukraine", "the democratic republic of the congo", "the republic of congo", "the islamic republic of iran", "the democratic people's republic of north korea", "the republic of south korea", "the people's democratic republic of lao", "the federal states of micronesia", "the republic of moldova", "the northern mariana islands", "the slovak republic", "the syrian arab republic", "the virgin islands", "the vatican", "the united arab emirates", "the central african republic", "the united republic of tanzania", "the maldives", "the marshall Islands", "the northern mariana islands", "the turks and caicos islands", "the democratic republic", "the russian federation"]}""")['the_countries']

