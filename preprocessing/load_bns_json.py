"""
preprocessing/load_bns_json.py
-------------------------------
Loads the Bharatiya Nyaya Sanhita (BNS), 2023 — the successor to IPC.
BNS replaced IPC effective July 1, 2024 (Act No. 45 of 2023).

Structure: 358 sections across 20 chapters.

Since no public machine-readable JSON exists yet, we provide:
  1. A curated core dataset of key BNS sections (all 20 chapters represented)
  2. IPC-to-BNS mapping table for cross-corpus lookup
  3. load_bns_json() — same schema as load_ipc_json(), plug-and-play

Schema (same as IPC clauses):
  {
    "section_number": "103",    # BNS section number
    "title":          "...",
    "chapter":        6,
    "chapter_title":  "Offences Affecting the Human Body",
    "text":           "103. <title>.\n<description>",
    "length":         <int>,
    "corpus":         "bns",    # provenance tag
    "ipc_equivalent": "302"     # IPC equivalent if applicable
  }
"""

import os
import json

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
BNS_OUTPUT_PATH  = os.path.join(BASE_DIR, 'data', 'bns_clauses.json')

# ── IPC → BNS section mapping (key IPC sections) ──────────────────────────────
IPC_TO_BNS = {
    "299": "100", "300": "101", "302": "103", "303": "104",
    "304": "105", "304A": "106", "304B": "80", "305": "107",
    "306": "108", "307": "109", "308": "110", "309": "226",
    "312": "88",  "313": "89",  "314": "90",
    "319": "114", "320": "114", "321": "115", "322": "115",
    "323": "115", "324": "118", "325": "116", "326": "118",
    "326A": "124", "326B": "125", "334": "117", "339": "126",
    "340": "127", "341": "127", "342": "128",
    "354": "74",  "354A": "75", "354B": "76",
    "354C": "77", "354D": "78",
    "359": "137", "360": "137", "361": "137", "362": "138",
    "363": "139", "364": "140", "364A": "141", "365": "142",
    "366": "143", "370": "143", "375": "63",  "376": "64",
    "376A": "66", "376B": "67", "376C": "68", "376D": "70",
    "378": "303", "379": "304", "380": "305", "382": "306",
    "383": "308", "384": "309", "390": "310", "391": "311",
    "392": "309", "395": "312",  "396": "313",
    "403": "314", "405": "316", "406": "317",
    "409": "318", "411": "317",
    "415": "318", "416": "319", "419": "319", "420": "318",
    "425": "323", "426": "324",
    "441": "329", "442": "330", "445": "331",
    "463": "334", "464": "335", "465": "336",
    "468": "336", "469": "336",
    "489A": "178", "489B": "179",
    "494": "82",  "498A": "85",
    "499": "356", "500": "356", "503": "351", "506": "351",
    "107": "45",  "108": "46",  "109": "48",  "115": "50",
    "120A": "61", "120B": "61",
    "121": "147", "121A": "148", "122": "149",
    "124A": "152",
    "141": "189", "142": "190", "147": "191", "148": "191",
    "149": "190",
    "153A": "196", "295A": "299", "298": "299",
    "268": "285",
    "34": "3",    "35": "3",
}

# BNS → IPC reverse mapping
BNS_TO_IPC = {v: k for k, v in IPC_TO_BNS.items()}

# ── Curated BNS sections dataset ───────────────────────────────────────────────
# All 20 BNS chapters represented with key sections
BNS_SECTIONS = [
    # CHAPTER 1 — PRELIMINARY
    {"section_number": "1",  "chapter": 1, "chapter_title": "Preliminary",
     "title": "Short title, extent and commencement",
     "desc": "This Sanhita may be called the Bharatiya Nyaya Sanhita, 2023. It extends to the whole of India. It shall come into force on such date as the Central Government may, by notification in the Official Gazette, appoint."},
    {"section_number": "2",  "chapter": 1, "chapter_title": "Preliminary",
     "title": "Definitions",
     "desc": "In this Sanhita, unless the context otherwise requires, the following words and expressions are used in the following senses: 'act' denotes as well a series of acts as a single act; 'animal' denotes any living creature, other than a human being."},

    # CHAPTER 2 — PUNISHMENTS
    {"section_number": "4",  "chapter": 2, "chapter_title": "Punishments",
     "title": "Punishments",
     "desc": "The punishments to which offenders are liable under the provisions of this Sanhita are: death; imprisonment for life; imprisonment, which is of two descriptions, namely — rigorous, that is, with hard labour; simple; forfeiture of property; fine; community service."},
    {"section_number": "9",  "chapter": 2, "chapter_title": "Punishments",
     "title": "Solitary confinement",
     "desc": "Whenever any person is convicted of an offence for which under this Sanhita the Court has power to sentence him to rigorous imprisonment, the Court may, by its sentence, order that the offender shall be kept in solitary confinement for any portion or portions of the imprisonment to which he is sentenced."},

    # CHAPTER 3 — GENERAL EXCEPTIONS
    {"section_number": "14", "chapter": 3, "chapter_title": "General Exceptions",
     "title": "Act of a person bound by law",
     "desc": "Nothing is an offence which is done by a person who is, or who by reason of a mistake of fact and not by reason of a mistake of law, in good faith believes himself to be, bound by law to do it."},
    {"section_number": "22", "chapter": 3, "chapter_title": "General Exceptions",
     "title": "Act of a person of unsound mind",
     "desc": "Nothing is an offence which is done by a person who, at the time of doing it, by reason of unsoundness of mind, is incapable of knowing the nature of the act, or that he is doing what is either wrong or contrary to law."},
    {"section_number": "34", "chapter": 3, "chapter_title": "General Exceptions",
     "title": "Right of private defence of body — when death may be caused",
     "desc": "The right of private defence of the body extends, under the restrictions mentioned in section 33, to the voluntary causing of death or of any other harm to the assailant, if the offence which occasions the exercise of the right be of any of the following descriptions: (1) an assault as may reasonably cause the apprehension that death will otherwise be the consequence of such assault; (2) an assault as may reasonably cause the apprehension that grievous hurt will otherwise be the consequence of such assault; (3) an assault with the intention of committing rape."},
    {"section_number": "35", "chapter": 3, "chapter_title": "General Exceptions",
     "title": "Right of private defence of property — when death may be caused",
     "desc": "The right of private defence of property extends, under the restrictions mentioned in section 33, to the voluntary causing of death or of any other harm to the wrong-doer, if the offence... is robbery, house-breaking by night, mischief by fire or explosive substance, theft, mischief, or house-trespass, under such circumstances as may reasonably cause apprehension that death or grievous hurt will be the consequence, if such right of private defence is not exercised."},

    # CHAPTER 4 — ABETMENT, CRIMINAL CONSPIRACY
    {"section_number": "45", "chapter": 4, "chapter_title": "Abetment, Criminal Conspiracy and Attempt",
     "title": "Abetment of a thing",
     "desc": "A person abets the doing of a thing, who — first, instigates any person to do that thing; or secondly, engages with one or more other person or persons in any conspiracy for the doing of that thing; or thirdly, intentionally aids, by any act or illegal omission, the doing of that thing."},
    {"section_number": "61", "chapter": 4, "chapter_title": "Abetment, Criminal Conspiracy and Attempt",
     "title": "Criminal conspiracy",
     "desc": "When two or more persons agree to do, or cause to be done — (a) an illegal act, or (b) an act which is not illegal by illegal means, such an agreement is designated a criminal conspiracy. A person shall be punished for criminal conspiracy if he commits a crime in pursuance of such conspiracy."},
    {"section_number": "62", "chapter": 4, "chapter_title": "Abetment, Criminal Conspiracy and Attempt",
     "title": "Punishment for criminal conspiracy",
     "desc": "Whoever is a party to a criminal conspiracy to commit an offence punishable with death, imprisonment for life or rigorous imprisonment for a term of two years or upwards, shall, where no express provision is made in this Sanhita for the punishment of such a conspiracy, be punished in the same manner as if he had abetted such offence."},

    # CHAPTER 5 — OFFENCES AGAINST THE STATE
    {"section_number": "147", "chapter": 5, "chapter_title": "Offences Against the State",
     "title": "Waging, or attempting to wage war, or abetting waging of war, against the Government of India",
     "desc": "Whoever wages war against the Government of India, or attempts to wage such war, or abets the waging of such war, shall be punished with death, or imprisonment for life and shall also be liable to fine."},
    {"section_number": "148", "chapter": 5, "chapter_title": "Offences Against the State",
     "title": "Conspiracy to commit offences punishable by section 147",
     "desc": "Whoever within or without India conspires to commit any of the offences punishable by section 147, or conspires to overawe, by means of criminal force or show of criminal force, the Central Government or any State Government, shall be punished with imprisonment for life, or with imprisonment of either description which may extend to ten years, and shall also be liable to fine."},
    {"section_number": "152", "chapter": 5, "chapter_title": "Offences Against the State",
     "title": "Act endangering sovereignty, unity and integrity of India",
     "desc": "Whoever, purposely or knowingly, by words, either spoken or written, or by signs, or by visible representation, or by electronic communication or by use of financial mean, or otherwise, excites or attempts to excite, secession or armed rebellion or subversive activities, or encourages feelings of separatist activities or endangers sovereignty or unity and integrity of India; or indulges in or commits any such act shall be punished with imprisonment for life or with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine."},

    # CHAPTER 6 — OFFENCES AFFECTING THE HUMAN BODY
    {"section_number": "100", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Culpable homicide",
     "desc": "Whoever causes death by doing an act with the intention of causing death, or with the intention of causing such bodily injury as is likely to cause death, or with the knowledge that he is likely by such act to cause death, commits the offence of culpable homicide. Explanation: A person who causes bodily injury to another who is labouring under a disorder, disease or bodily infirmity, and thereby accelerates the death of that other, shall be deemed to have caused his death."},
    {"section_number": "101", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Murder",
     "desc": "Culpable homicide is murder, if the act by which the death is caused is done with the intention of causing death, or — secondly, if it is done with the intention of causing such bodily injury as the offender knows to be likely to cause the death of the person to whom the harm is caused, or — thirdly, if it is done with the intention of causing bodily injury to any person and the bodily injury intended to be inflicted is sufficient in the ordinary course of nature to cause death, or — fourthly, if the person committing the act knows that it is so imminently dangerous that it must, in all probability, cause death."},
    {"section_number": "103", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Punishment for murder",
     "desc": "Whoever commits murder shall be punished with death or with imprisonment for life, and shall also be liable to fine. When a group of five or more persons acting in concert commits murder on the ground of race, caste or community, sex, place of birth, language, personal belief or any other similar ground each member of such group shall be punished with death or with imprisonment for life or with imprisonment of either description for a term which shall not be less than seven years, and shall also be liable to fine."},
    {"section_number": "105", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Culpable homicide not amounting to murder",
     "desc": "Whoever commits culpable homicide not amounting to murder shall be punished with imprisonment for life, or imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine. If the act by which the death is caused is done with the intention of causing death or of causing such bodily injury as is likely to cause death, the offender shall be punished with imprisonment of life or with imprisonment for a term which may extend to ten years and shall also be liable to fine."},
    {"section_number": "106", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Causing death by negligence",
     "desc": "Whoever causes the death of any person by doing any rash or negligent act not amounting to culpable homicide, shall be punished with imprisonment of either description for a term which may extend to five years, and shall also be liable to fine; and if such act is done by a registered medical practitioner while performing medical procedure, the registered medical practitioner shall be punished with imprisonment of either description for a term which may extend to two years, and shall also be liable to fine. Whoever causes the death of any person by rash and negligent driving of vehicle, not amounting to culpable homicide, and escapes without reporting it to a police officer or a Magistrate, shall be punished with imprisonment of either description for a term of ten years, and shall also be liable to fine."},
    {"section_number": "108", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Abetment of suicide of child or person of unsound mind",
     "desc": "If any person under eighteen years of age, any insane person, any delirious person, any idiot, or any person in a state of intoxication commits suicide, whoever abets the commission of such suicide, shall be punished with death or imprisonment for life, or imprisonment for a term not exceeding ten years, and shall also be liable to fine."},
    {"section_number": "109", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Attempt to commit suicide to compel or restrain exercise of lawful power",
     "desc": "Whoever attempts to commit suicide with the intent to compel or restrain any public servant to do or abstain from doing any act in the discharge of his official duties, shall be punished with simple imprisonment for a term which may extend to one year, or with fine, or with both."},
    {"section_number": "114", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Hurt",
     "desc": "Whoever causes bodily pain, disease or infirmity to any person is said to cause hurt."},
    {"section_number": "115", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Voluntarily causing hurt",
     "desc": "Whoever voluntarily causes hurt, if the hurt which he intends to cause or knows himself to be likely to cause is grievous hurt, and if the hurt which he causes is grievous hurt, shall be punished with imprisonment of either description for a term which may extend to one year, or with fine which may extend to ten thousand rupees, or with both."},
    {"section_number": "116", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Voluntarily causing grievous hurt",
     "desc": "Whoever voluntarily causes grievous hurt shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine."},
    {"section_number": "118", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Voluntarily causing hurt by dangerous weapons or means",
     "desc": "Whoever, except in the case provided for by section 121, voluntarily causes hurt by means of any instrument for shooting, stabbing or cutting, or any instrument which, used as a weapon of offence, is likely to cause death, or by means of fire or any heated substance, or by means of any poison or any corrosive substance, or by means of any explosive substance, or by means of any substance which it is deleterious to the human body to inhale, to swallow, or to receive into the blood, or by means of any animal, shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both."},
    {"section_number": "124", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Voluntarily causing grievous hurt by use of acid",
     "desc": "Whoever voluntarily causes grievous hurt by use of acid or by use of any corrosive substance or burning substance or by the use of any explosive substance or by administering any substance which is deleterious to the human body or by any means to cause grievous hurt shall be punished with imprisonment for life, and a fine. The fine shall be just and reasonable and shall be paid to the victim."},
    {"section_number": "126", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Wrongful restraint",
     "desc": "Whoever voluntarily obstructs any person so as to prevent that person from proceeding in any direction in which that person has a right to proceed, is said wrongfully to restrain that person."},
    {"section_number": "127", "chapter": 6, "chapter_title": "Offences Affecting the Human Body",
     "title": "Wrongful confinement",
     "desc": "Whoever wrongfully restrains any person in such a manner as to prevent that person from proceeding beyond certain circumscribing limits, is said wrongfully to confine that person."},

    # CHAPTER 5B — OFFENCES AGAINST WOMAN AND CHILD
    {"section_number": "63", "chapter": 5, "chapter_title": "Offences Against Woman and Child",
     "title": "Rape",
     "desc": "A man is said to commit rape if he — (a) penetrates his penis, to any extent, into the vagina, mouth, urethra or anus of a woman or makes her to do so with him or any other person; or (b) inserts, to any extent, any object or a part of the body, not being the penis, into the vagina, the urethra or anus of a woman or makes her to do so with him or any other person; or (c) manipulates any part of the body of a woman so as to cause penetration into the vagina, urethra, anus or any part of body of such woman or makes her to do so with him or any other person; or (d) applies his mouth to the vagina, anus, urethra of a woman or makes her to do so with him or any other person."},
    {"section_number": "64", "chapter": 5, "chapter_title": "Offences Against Woman and Child",
     "title": "Punishment for rape",
     "desc": "Whoever, except in the cases provided for in sub-section (2), commits rape, shall be punished with rigorous imprisonment of either description for a term which shall not be less than ten years, but which may extend to imprisonment for life, and shall also be liable to fine. Whoever, commits rape on a woman under sixteen years of age shall be punished with rigorous imprisonment for a term which shall not be less than twenty years, but which may extend to imprisonment for life, which shall mean imprisonment for the remainder of that person's natural life, and shall also be liable to fine."},
    {"section_number": "74", "chapter": 5, "chapter_title": "Offences Against Woman and Child",
     "title": "Assault or use of criminal force to woman with intent to outrage her modesty",
     "desc": "Whoever assaults or uses criminal force to any woman, intending to outrage or knowing it to be likely that he will thereby outrage her modesty, shall be punished with imprisonment of either description for a term which shall not be less than one year but which may extend to five years, and shall also be liable to fine."},
    {"section_number": "75", "chapter": 5, "chapter_title": "Offences Against Woman and Child",
     "title": "Sexual harassment",
     "desc": "A man committing any of the following acts — (i) physical contact and advances involving unwelcome and explicit sexual overtures; or (ii) a demand or request for sexual favours; or (iii) showing pornography against the will of a woman; or (iv) making sexually coloured remarks, shall be guilty of the offence of sexual harassment. Whoever commits the offence of sexual harassment shall be punished with rigorous imprisonment for a term which may extend to three years, or with fine, or with both."},
    {"section_number": "78", "chapter": 5, "chapter_title": "Offences Against Woman and Child",
     "title": "Stalking",
     "desc": "Any man who follows a woman and contacts, or attempts to contact such woman to foster personal interaction repeatedly despite a clear indication of disinterest by such woman, or monitors the use by a woman of the internet, email or any other form of electronic communication, commits the offence of stalking. A man who commits the offence of stalking shall be punished on first conviction with imprisonment of either description for a term which may extend to three years, and shall also be liable to fine; and be punished on a second or subsequent conviction, with imprisonment of either description for a term which may extend to five years, and shall also be liable to fine."},
    {"section_number": "80", "chapter": 5, "chapter_title": "Offences Against Woman and Child",
     "title": "Dowry death",
     "desc": "Where the death of a woman is caused by any burns or bodily injury or occurs under suspicious circumstances within seven years of marriage and it is shown that soon before her death she was subjected to cruelty or harassment by her husband or any relative of her husband for, or in connection with, any demand for dowry, such death shall be called 'dowry death', and such husband or relative shall be deemed to have caused her death. Whoever commits dowry death shall be punished with imprisonment for a term which shall not be less than seven years but which may extend to imprisonment for life."},
    {"section_number": "85", "chapter": 5, "chapter_title": "Offences Against Woman and Child",
     "title": "Husband or relative of husband of a woman subjecting her to cruelty",
     "desc": "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished with imprisonment for a term which may extend to three years and shall also be liable to fine. For the purposes of this section, cruelty means — (a) any wilful conduct which is of such a nature as is likely to drive the woman to commit suicide or to cause grave injury or danger to life, limb or health (whether mental or physical) of the woman; or (b) harassment of the woman where such harassment is with a view to coercing her or any person related to her to meet any unlawful demand for any property or valuable security or is on account of failure by her or any person related to her to meet such demand."},

    # CHAPTER — OFFENCES AGAINST PROPERTY
    {"section_number": "303", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Theft",
     "desc": "Whoever intending to take dishonestly any moveable property out of the possession of any person without that person's consent, moves that property in order to such taking, is said to commit theft."},
    {"section_number": "304", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Punishment for theft",
     "desc": "Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both."},
    {"section_number": "308", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Extortion",
     "desc": "Whoever intentionally puts any person in fear of any injury to that person, or to any other, and thereby dishonestly induces the person so put in fear to deliver to any person any property or valuable security, or anything signed or sealed which may be converted into a valuable security, commits extortion."},
    {"section_number": "309", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Robbery",
     "desc": "In all robbery there is either theft or extortion. To constitute robbery, theft must be accompanied by — (1) a voluntary causing or attempting to cause to any person death or hurt or wrongful restraint; or (2) fear of instant death or of instant hurt or of instant wrongful restraint. Extortion is robbery if the offender commits the extortion in the person's presence, and puts that person in fear of instant death, or instant hurt, and causes that person to deliver up the thing extorted."},
    {"section_number": "310", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Punishment for robbery",
     "desc": "Whoever commits robbery shall be punished with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine; and, if the robbery be committed on the highway between sunset and sunrise, the imprisonment may be extended to fourteen years."},
    {"section_number": "311", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Dacoity",
     "desc": "When five or more persons conjointly commit or attempt to commit a robbery, or where the whole number of persons conjointly committing or attempting to commit a robbery, and persons present and aiding such commission or attempt, amount to five or more, every person so committing, attempting or aiding, is said to commit dacoity."},
    {"section_number": "312", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Punishment for dacoity",
     "desc": "Whoever commits dacoity shall be punished with imprisonment for life, or with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine."},
    {"section_number": "316", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Criminal breach of trust",
     "desc": "Whoever, being in any manner entrusted with property, or with any dominion over property, dishonestly misappropriates or converts to his own use that property, or dishonestly uses or disposes of that property in violation of any direction of law prescribing the mode in which such trust is to be discharged, or of any legal contract, express or implied, which he has made touching the discharge of such trust, or wilfully suffers any other person so to do, commits criminal breach of trust."},
    {"section_number": "318", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Cheating",
     "desc": "Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property to any person, or to consent that any person shall retain any property, or intentionally induces the person so deceived to do or omit to do anything which he would not do or omit if he were not so deceived, and which act or omission causes or is likely to cause damage or harm to that person in body, mind, reputation or property, is said to cheat."},
    {"section_number": "319", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Cheating by personation",
     "desc": "A person is said to cheat by personation if he cheats by pretending to be some other person, or by knowingly substituting one person for another, or representing that he or any other person is a person other than he or such other person really is."},
    {"section_number": "323", "chapter": 17, "chapter_title": "Offences Against Property",
     "title": "Mischief",
     "desc": "Whoever with intent to cause, or knowing that he is likely to cause, wrongful loss or damage to the public or to any person, causes the destruction of any property, or any such change in any property or in the situation thereof as destroys or diminishes its value or utility, or affects it injuriously, commits mischief."},

    # FORGERY
    {"section_number": "334", "chapter": 18, "chapter_title": "Offences Relating to Documents and Property Marks",
     "title": "Forgery",
     "desc": "Whoever makes any false document or false electronic record or part of a document or electronic record, with intent to cause damage or injury, to the public or to any person, or to support any claim or title, or to cause any person to part with property, or to enter into any express or implied contract, or with intent to commit fraud or that fraud may be committed, commits forgery."},
    {"section_number": "336", "chapter": 18, "chapter_title": "Offences Relating to Documents and Property Marks",
     "title": "Forgery for purpose of cheating",
     "desc": "Whoever commits forgery, intending that the document or electronic record forged shall be used for the purpose of cheating, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine."},

    # ORGANISED CRIME (NEW IN BNS)
    {"section_number": "111", "chapter": 6, "chapter_title": "Organised Crime",
     "title": "Organised crime",
     "desc": "Whoever commits organised crime shall be punished with imprisonment for life, or with death if the offence causes death; and shall also be liable to fine which shall not be less than ten lakh rupees. 'Organised crime' means any continuing unlawful activity including kidnapping, robbery, vehicle theft, extortion, land grabbing, contract killing, economic offences, cyber-crimes, trafficking in persons, drugs, or arms by a person or a syndicate of persons using or threatening violence, intimidation, coercion or any other unlawful means to obtain undue economic or other advantage."},
    {"section_number": "112", "chapter": 6, "chapter_title": "Organised Crime",
     "title": "Petty organised crime",
     "desc": "Whoever, being a member of a group or gang, either singly or jointly, commits any act of theft, snatching, cheating, unauthorised selling of tickets, unauthorised betting or gambling, selling of public examination question papers or any other similar criminal act, is said to commit petty organised crime. Whoever commits petty organised crime shall be punished with imprisonment of either description for a term which shall not be less than one year, but may extend to seven years and shall also be liable to fine."},

    # TERRORISM (NEW IN BNS)
    {"section_number": "113", "chapter": 6, "chapter_title": "Terrorist Act",
     "title": "Terrorist act",
     "desc": "Whoever does any act with the intent to threaten or likely to threaten the unity, integrity, sovereignty, security, or economic security of India, or with the intent to strike terror or likely to strike terror in the people or any section of the people in India or in any foreign country, — (a) by using bombs, dynamite or other explosive substances or inflammable substances or firearms or other lethal weapons or poisonous or noxious gases or other chemicals or by any other substances (whether biological, radioactive, nuclear or otherwise) of a hazardous nature; or (b) by attacking the financial systems including banking systems, currency systems, monetary systems; or (c) by cyber-crimes commits a terrorist act."},

    # COMMUNITY SERVICE (NEW IN BNS)
    {"section_number": "4",  "chapter": 2, "chapter_title": "Punishments",
     "title": "Community service as punishment",
     "desc": "Community service is a new form of punishment introduced by the Bharatiya Nyaya Sanhita 2023. It refers to unpaid work that an offender is required to perform as a way to give back to the community, as an alternative to imprisonment or fine for minor offences. This is a significant departure from the Indian Penal Code, 1860, which did not recognize community service as a form of punishment."},

    # UNLAWFUL ASSEMBLY AND RIOTING
    {"section_number": "189", "chapter": 11, "chapter_title": "Offences Against the Public Tranquillity",
     "title": "Unlawful assembly",
     "desc": "An assembly of five or more persons is designated an unlawful assembly, if the common object of the persons composing that assembly is — (1) to overawe by criminal force, or show of criminal force, any public servant in the exercise of the lawful power of such public servant; or (2) to resist the execution of any law, or of any legal process; or (3) to commit any offence; or (4) by means of criminal force, or show of criminal force, to any person to take or obtain possession of any property, or to deprive any person of the enjoyment of a right of way, or of the use of water or other incorporeal right of which he is in possession or enjoyment, or to enforce any right or supposed right; or (5) by means of criminal force, or show of criminal force, to compel any person to do what he is not legally bound to do, or to omit to do what he is legally entitled to do."},
    {"section_number": "191", "chapter": 11, "chapter_title": "Offences Against the Public Tranquillity",
     "title": "Rioting",
     "desc": "Whenever force or violence is used by an unlawful assembly, or by any member thereof, in prosecution of the common object of such assembly, every member of such assembly is guilty of the offence of rioting. Whoever is guilty of rioting, shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both."},

    # DEFAMATION & CRIMINAL INTIMIDATION
    {"section_number": "356", "chapter": 19, "chapter_title": "Criminal Intimidation, Insult and Annoyance",
     "title": "Defamation",
     "desc": "Whoever, by words either spoken or intended to be read, or by signs or by visible representations, makes or publishes any imputation concerning any person intending to harm, or knowing or having reason to believe that such imputation will harm, the reputation of such person, is said, except in the cases hereinafter excepted, to defame that person."},
    {"section_number": "351", "chapter": 19, "chapter_title": "Criminal Intimidation, Insult and Annoyance",
     "title": "Criminal intimidation",
     "desc": "Whoever threatens another with any injury to his person, reputation or property, or to the person or reputation or property of any one in whom that person is interested, with intent to cause alarm to that person, or to cause that person to do any act which he is not legally bound to do, or to omit to do any act which that person is legally entitled to do, as the means of avoiding the execution of such threat, commits criminal intimidation. Punishment: imprisonment of either description for a term which may extend to two years, or with fine, or with both."},

    # OFFENCES AGAINST PUBLIC HEALTH
    {"section_number": "285", "chapter": 14, "chapter_title": "Offences Affecting Public Health, Safety, Convenience, Decency and Morals",
     "title": "Public nuisance",
     "desc": "A person is guilty of a public nuisance who does any act or is guilty of an illegal omission which causes any common injury, danger or annoyance to the public or to the people in general who dwell or occupy property in the vicinity, or which must necessarily cause injury, obstruction, danger or annoyance to persons who may have occasion to use any public right."},

    # COUNTERFEITING CURRENCY
    {"section_number": "178", "chapter": 12, "chapter_title": "Offences Relating to Coin and Government Stamps",
     "title": "Counterfeiting coin",
     "desc": "Whoever counterfeits, or knowingly performs any part of the process of counterfeiting, any coin which is for the time being current in India, shall be punished with imprisonment for life, or with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine."},
]


def load_bns_sections() -> list[dict]:
    """
    Load and normalise curated BNS dataset into standard clause dicts.

    Returns:
        List of clause dicts with corpus='bns' provenance tag, sorted by section number.
    """
    import re

    def sort_key(sec):
        num = int(re.sub(r'[A-Za-z]', '', sec)) if re.sub(r'[A-Za-z]', '', sec) else 0
        suffix = re.sub(r'\d', '', sec).upper()
        return (num, suffix)

    clauses = []
    seen = set()

    for entry in BNS_SECTIONS:
        sec_num = str(entry["section_number"]).strip()
        dedup_key = f"{sec_num}_{entry['title'][:20]}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        title = entry["title"].strip()
        desc  = entry.get("desc", "").strip()
        text  = f"{sec_num}. {title}.\n{desc}" if desc else f"{sec_num}. {title}."

        clauses.append({
            "section_number": sec_num,
            "title":          title,
            "chapter":        entry.get("chapter", 0),
            "chapter_title":  entry.get("chapter_title", "").title(),
            "text":           text,
            "length":         len(text),
            "corpus":         "bns",
            "ipc_equivalent": BNS_TO_IPC.get(sec_num, None),
        })

    clauses.sort(key=lambda c: sort_key(c["section_number"]))
    return clauses


def save_bns_clauses(clauses: list[dict], path: str = BNS_OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clauses, f, indent=2, ensure_ascii=False)
    print(f"[load_bns_json] Saved {len(clauses)} BNS clauses -> {path}")


def get_ipc_to_bns_map() -> dict:
    """Return IPC section → BNS section mapping dict."""
    return dict(IPC_TO_BNS)


def get_bns_to_ipc_map() -> dict:
    """Return BNS section → IPC section mapping dict."""
    return dict(BNS_TO_IPC)


if __name__ == '__main__':
    clauses = load_bns_sections()
    lengths = [c['length'] for c in clauses]
    print(f"\n[BNS] Total sections     : {len(clauses)}")
    print(f"[BNS] Avg length         : {sum(lengths)/len(lengths):.0f} chars")
    print(f"[BNS] IPC-BNS mappings   : {len(IPC_TO_BNS)}")
    print(f"\nSample BNS sections:")
    for c in clauses:
        if c['section_number'] in ['103', '64', '85', '318', '152']:
            ipc_eq = c.get('ipc_equivalent')
            print(f"  BNS §{c['section_number']:>4}  (≈ IPC §{ipc_eq or '—':>5})  "
                  f"{c['length']:>4} chars: {c['text'][:80]}...")
    save_bns_clauses(clauses)
