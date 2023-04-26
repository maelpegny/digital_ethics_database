# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 07:52:26 2023

@author: maelp
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:28:41 2023

@author: maelp
"""


# import necessary packages
import pandas as pd
import re
from bs4 import BeautifulSoup
from googletrans import Translator

# Convert h5 files of webscraped data into pandas dataframe
osnabrueck_df = pd.read_hdf(
    'C:/fake_file_path\osnabrueck.h5',
    key="webscrap")
bochum_cais_df = pd.read_hdf(
    'C:/fake_file_path/bochum_cais.h5',
    key='webscrap')
berlin_wzb_df = pd.read_hdf(
    'C:/fake_file_path/berlin_wzb.h5', 
    key='webscrap')
hamburg_df = pd.read_hdf(
    'C:/fake_file_path\hamburg.h5',
    key='webscrap')
hamburg_hbi_df = pd.read_hdf(
    'C:/fake_file_path\hamburg_hbi.h5', 
    key='webscrap')
tuebingen_izew_df = pd.read_csv(
    'C:/fake_file_path/tuebingen_izew.csv')
tum_digital_services_df = pd.read_hdf(
    'C:/fake_file_path/tum_digital_services.h5',
    key='webscrap')
tum_ieai_leadership_df = pd.read_hdf(
    'C:/fake_file_path/tum_leadership.h5',
    key='webscrap')
tum_ieai_associates_df = pd.read_hdf(
    'C:/fake_file_path/tum_ieai_associates.h5', 
    key='webscrap')
manual_germany_df = pd.read_csv(
    'C:/fake_file_path/manual_germany.csv',
    encoding='utf-8')
bias_hannover_df = pd.read_csv(
    'C:/fake_file_path/bias_hannover.csv', 
    encoding= 'utf-8') 
duesseldorf_iid_df = pd.read_csv(
    'C:/fake_file_path\iid.csv',
    encoding='utf-8')
muenchen_bidt_df = pd.read_csv(
    'C:/fake_file_path\muenchen_bidt.csv',
    encoding='utf-8')
koeln_df = pd.read_csv(
    'C:/fake_file_path/koeln.csv', 
    encoding='utf-8')
ide_leitung_df = pd.read_csv(
    'C:/fake_file_path\stuttgart_ide_leitung.csv',
    encoding='utf-8')
ide_mitarbeiter_df = pd.read_csv(
    'C:/fake_file_path\stuttgart_ide_mitarbeiter.csv',
    encoding='utf-8')

# Concatenating list of dataframes into one dataframe for Germany
germany_df = pd.concat([osnabrueck_df, bochum_cais_df, berlin_wzb_df,
                        hamburg_df, hamburg_hbi_df,
                        tuebingen_izew_df,
                        tum_ieai_leadership_df, tum_ieai_associates_df,
                        manual_germany_df,
                        bias_hannover_df, duesseldorf_iid_df, muenchen_bidt_df,
                        koeln_df,
                        ide_leitung_df, ide_mitarbeiter_df],
                       ignore_index=True)


# Forcing string types on all columns of dataframe
germany_df = germany_df.astype('string')

#Replace missing values with "Not available"
germany_df.fillna('Not available', inplace=True)


# list of symbols to eliminate from entire dataframe
germany_df_clean = germany_df.replace(
    to_replace=r'(\\n|\\xa0|\\t|\[|\]|\'|\\u200b|\\u202f|\\xad|\\u2028)',
    value='', regex=True)

print(germany_df_clean.dtypes)

# Clean HTML tags from text of columns "Email", "Description", "CV",
# "Research topics"
list_column_with_html = ['Email', 'Description',
                         'CV', 'Research Topics', 'Adress & Email']
for name in list_column_with_html:
    for key, value in germany_df_clean[name].iteritems():
        germany_df_clean[name].loc[key] = BeautifulSoup(value).get_text()


# SEPARATE NAME AND ACADEMIC TITLE

# List of academic titles to loop over
list_academic_titles = ["Dipl.-Wirtschaftsjuristin", "Dr. Dipl.-Psych.", 
                        "Prof. Dr. Dr. h.c.",
                        "Dipl.-Germ. Univ.", "Maître en droit",
                        "LL.M., Maître en Droit",
                        "Univ.-Prof. Dr.",
                        "Dipl.-Päd.", 'Dipl. -Soz.',
                        " LL.M. \(Berkeley\)", "\(Berkeley\)",
                        "LL.M. \(Harvard\)",
                        "Dr. habil.",
                        "Pr. Dr. Dr.", "Dipl.-Kffr.", "MSc \(LSE\)","Pr. iur.",
                        "(vice speaker)",  "Prof. Dr.",  "PD Dr.", "Jun.-Pr."
                        "LL.M.", "M. Sc.", "M. Sc", "MSc.", "M.Sc.", "Ph.D.",
                        "Dr.", 
                        "M. A.",
                        "M A", "M.A.", "Pr."]


# Find academic titles in "Name and academic title" column and write them
# in the 'Academic title' column

for key, value in germany_df_clean['Name and academic title'].iteritems():
    for substring in list_academic_titles:
        if re.search(substring, str(value)):
            germany_df_clean['Academic title'][key] = substring

# Clean "Name and academic title" column from academic titles
for substring in list_academic_titles:
    germany_df_clean['Name and academic title'] = \
        germany_df_clean['Name and academic title'].str.replace(
        substring, '')


# Eliminate initial whitespace with regular expression
germany_df_clean['Name and academic title'] = \
    germany_df_clean['Name and academic title'].replace(
    to_replace=r'\^s+', value='', regex=True)

# Eliminate some strings in "Name and academic title" column
list_name_strings_to_eliminate = [",", "Translate to Englisch:"]
for substr in list_name_strings_to_eliminate:
    germany_df_clean['Name and academic title'] = \
    germany_df_clean['Name and academic title'].str.replace(
        substr, '')

# Rectifiy name of Professor d'Aquin
germany_df_clean['Name and academic title'] = \
germany_df_clean['Name and academic title'].str.replace(
    "dAquin", "d'Aquin")

# Eliminate all whitespaces from Email column via regular expressions
germany_df_clean['Email'] = germany_df_clean['Email'].replace(
    to_replace=r'\s+', value='', regex=True)

# List of strings to eliminate from Email column
list_email_strings_to_eliminate = ['request', 'Email:', 'E-Mail:',
                                   'info@cais-research.de', ',',
                                   'ProfileatTUM', " ", "spamprevention"]
# Loop over values in list to eliminate strings from
for substr in list_email_strings_to_eliminate:
    germany_df_clean['Email'] = germany_df_clean['Email'].str.replace(
        substr, '')

# Replace a list of strings by the @ sign in the "Email" column
list_email_strings_to_replace = [' /a/ ', '\(at\)', '\(a\) ', '/at/', ]
for substr in list_email_strings_to_replace:
    germany_df_clean['Email'] = germany_df_clean['Email'].str.replace(
        substr, '@')

# Replace the "atwzb" in Weizenbaum email addresses by "@wzb
germany_df_clean["Email"] = germany_df_clean['Email'].str.replace(
    'atwzb', '@wzb')
# Suppress double writing of Weizenbaum Institute addresses
for key, value in germany_df_clean['Email'].iteritems():
    text = value
    head, sep, tail = text.partition('.eu')
    germany_df_clean['Email'][key] = head + sep

# CLEAN OUT INDIVIDUALS WHO ARE NOT RESEARCHERS

# Specify list of non-researchers roles to iterate over in the next step
list_of_non_academic = ['Human Resources', 'Team Assistant', 
                        'Research Transfer and Events', 
                        'Purchasing Manager', 
                        'Coordinator for Research Information and Scientific Controlling',
                        'Beauftragte für Forschungsmanagement des Schwerpunkts',
                        'Research and Event Coordinator', 
                        'Head of Science Communication',
                        'Purchasing Manager', 
                        'Back Office and Editorial Team of M&K', 
                        'Executive committee', 'Managing office', 
                        'Student Assistant',
                        'Secretariat & Purchasing', 'Human resources', 
                        'System Administrator', 
                        'Apprentice in Office Management', 'Personnel', 
                        'Third-Party Funding and Budget Management', 
                        'Management Assistant',
                        'Management Assistent', 'Library', 'IT', 'Secretariat', 
                        'Science Communication / Podcast', 
                        'Presse und Öffentlichkeitsarbeit', 
                        'Referentin für Wissenschaftskommunikation', 
                        'Office of the Research Manager', 
                        'Wissenschaftliche Hilfskraft', 
                        'Studentische Hilfskraft']


# List of relevant column names for this operation
list_column_names = ['Role in team', 'Name and academic title']

# Cleaning out non-researchers
for name in list_column_names:
    for key, value in germany_df_clean[name].iteritems():
        for substring in list_of_non_academic:
            if re.search(substring, str(value)):
                germany_df_clean.drop(
                    germany_df_clean.loc[germany_df_clean.index == key].index,
                    axis=0, inplace=True)


# List of irrelevant individuals
list_non_scholars = ['names erased to preserve privacy']

# List of irrelevant researchers
list_irrelevant_scholars = ['names erased to preserve privacy']

# List of projects which got scraped instead of names
list_projects = ['Ethics and Education', 'Research Focus Media Ethics', 
                 'Public welfare orientation in the age of digitalization',
                 'Society Culture and Technological Change','Forum Privatheit',
                 'Forum vatheit', 'Leadership Ethics', 
                 'Theoretical Foundation of Organizational Ethics',
                 'ELISA', 'VIKING', 'Research Focus', 'migsst', 
                 'Integrated civil security', 'The joint project', 'ESTER',
                 'aKtIv', 'ZisSch', 'Public Welfare Orientation', 'KOPHIS', 
                 'PREVENT', 'IDeA', 'Nature and Sustainable Development', 
                 'WeNet', 'PODESTA', 'BATATA', "digilog@bw"]

# Concatenate previous lists in a list of irrelevant values
list_irrelevant_values = list_non_scholars + \
    list_irrelevant_scholars + list_projects + list_of_non_academic


# Drop any row where name is in list_of_irrelevant values
for key, value in germany_df_clean['Name and academic title'].iteritems():
    for substring in list_irrelevant_values:
        if re.search(substring, str(value)):
            germany_df_clean.drop(
                germany_df_clean.loc[germany_df_clean.index == key].index,
                axis=0, inplace=True)

# Clarify role in team by addding 'member of project "Project"'
germany_df_clean['Role in team'] = \
germany_df_clean['Role in team'].str.replace(
    'Society, Culture and Technological Change', 
    'Member of project "Society, Culture and Technological Change"')


# Erase useless punctuation marks left by html tag cleaning
germany_df_clean = germany_df_clean.replace(
    to_replace= r'^, +', value='', regex=True)
#Replace some superfluous symbols created by HTML tag cleaning
list_of_columns = ['Description', 'Research Topics', 'Career', 'CV']
list_strings_elim = [',.', ', .',',. ,', ',,', ', ,', ', "','", "', 
                     'Geschwister-Scholl-Platz 72074 Tübingen GermanyPhone:',
                     'Geschwister-Scholl-Platz 72074 Tübingen Germany Phone:',
                     '"Subscribe to our newsletter and',
                     'receive the Institutes latest news via email.',
                     'SUBSCRIBE!']

for name in list_of_columns:
    for string in list_strings_elim:
        germany_df_clean[name] = germany_df_clean[name].str.replace(string,
                                                                    ' ')

list_strings_replace = ['Short bio,']
for name in list_of_columns:
    for string in list_strings_replace:
        germany_df_clean[name] = \
        germany_df_clean[name].str.replace(string,'Short bio: ')
                                                                                                                                                         
# Instantiate Translator object from Googletrans package
translator = Translator()

# Translate "Description", "CV" and "Career" columns from German to English
germany_df_clean['Translated CV'] = germany_df_clean['CV'].apply(
    lambda x: translator.translate(x, dest='en').text)
germany_df_clean['Translated Description'] = \
    germany_df_clean['Description'].apply(
        lambda x: translator.translate(x, dest='en').text)
germany_df_clean['Translated Career'] = germany_df_clean['Career'].apply(
    lambda x: translator.translate(x, dest='en').text)
germany_df_clean['Translated Research Topics'] = \
    germany_df_clean['Research Topics'].apply(
    lambda x: translator.translate(x, dest='en').text)

# Clean out mistakes in "Translated Description"
germany_df_clean['Translated Description'] = \
    germany_df_clean['Translated Description'].str.replace('dr.', 'Dr.')
    
                              
# Reorder columns to facilitate readability
germany_df_clean = germany_df_clean[['Name and academic title',
                                     'Academic title', 'Gender',
                                     'Email',
                                     'address', 'Postal Address',
                                     'Coordinates', 'Adress & Email', 
                                     'Role in team',
                                     'Research Topics',
                                     'Translated Research Topics',
                                     'Research topics (classification for filters)',
                                     'description', 'Description', 
                                     'Translated Description', 'CV',
                                     'Translated CV', 'Career', 
                                     'Translated Career', 
                                     'Areas of Expertise',
                                     'Memberships', 'Former Memberships ', 
                                     'Current Memberships (Professional Associations, Committees, Boards, Non-Profit)',
                                     'Academic Service & Advisory Boards',
                                     'Editorial Boards', 'Executive Boards',
                                     'Current Research Projects on Digital Ethics',
                                     'Former Research Projects', 
                                     'Current & Completed Projects',
                                     'Projects', 'Research Group', 'category',
                                     'Awards',  'Field of study',
                                     'Former positions and visiting positions',
                                     'Academic institution (according to webpage used in data collection)',  'Center', 'Main position by theme', 'Main position by geography', 'Other academic positions', 'Previous non-academic work experience', 'Other non-academic professional positions', 'Notes', 'map_id', 'pic', 'icon', 'lat', 'lng', 'anim', 'infoopen', 'approved', 'retina']]
# Sort names in alphabetic order
germany_df_clean = germany_df_clean.sort_values(
    'Name and academic title', ascending=True)
# Drop superfluous columns
germany_df_clean = germany_df_clean.drop(['address', 'description'], axis=1)
germany_clean_csv = germany_df_clean.to_csv()
# Write csv file to computer
file_path = \
r'C:\fake_file_path\germany_clean.csv'
with open(file_path, 'w', encoding='utf8') as germany:
    germany.write(germany_clean_csv)


