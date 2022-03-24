import pandas as pd
import zipfile
import sqlite3
import os


####### SET THESE PARAMETERS #######

# File path for where the zipped Prosper listing and loan data folders are located.
file_loc = 'DATA LOCATION HERE'

# Import zipped Prosper files and create data tables? Requires valid file_loc.
create_db_tables_from_prosper = False

# Do you already have the joined data output and want to load into SQL directly?
# only set to True if joined_data does not exist or you want to overwrite
load_joined_from_csv = False

# join the model output (notches.csv?) to joined_data in the loan_database?
add_model_output = True

# create mini database to be included in the submission.
create_data_viz_tables = True

#######################################

#
# ----------------------------------------------------------------
# Bring data in from Prosper Zip files
# ----------------------------------------------------------------

def import_to_db(filepath = file_loc):
    '''
    :param filepath: string, location of zipped Prosper data files
    :return: None
    '''

    # assumes loan_database exists
    try:
        connection = sqlite3.connect("loan_database")
    except sqlite3.Error as error:
        print('Error occurred - ', error)

    for filename_zip in os.listdir(filepath):
        print("Processing: " + filename_zip)
        zf = zipfile.ZipFile(filepath + '/' + filename_zip)

        if filename_zip.find('Listing') >= 0:
            df = pd.read_csv(zf.open(zf.infolist()[0]), low_memory=False, encoding="ISO-8859-1")
            df = df[(df['listing_status_reason'] == 'Completed')]
            df.to_sql('listing', connection, if_exists='append')
        elif filename_zip.find('Loan') >= 0:
            df = pd.read_csv(zf.open(zf.infolist()[0]), low_memory=False)
            df = df[(df['loan_status_description'] == 'COMPLETED') | (df['loan_status_description'] == 'CHARGEOFF')]
            df.to_sql('loan', connection, if_exists='append')
        else:
            print('Missed File: ' + filename_zip)


def create_db_tables(output_csv = False, zipped_file_loc = None):
    '''
    :param output_csv: Boolean, output a CSV of the final table?
    :return: None
    '''

    # assumes loan_database exists
    try:
        connection = sqlite3.connect("loan_database")
    except sqlite3.Error as error:
        print('Error occurred - ', error)

    # import zipped files
    import_to_db(filepath = zipped_file_loc)

    index_sql = '''
        CREATE INDEX loan_index ON loan (loan_number, origination_date, amount_borrowed, prosper_rating)
    '''
    connection.execute(index_sql)

    index_sql = '''
            CREATE INDEX listing_index ON listing (listing_number, loan_origination_date, amount_funded, prosper_rating)
    '''
    connection.execute(index_sql)

    sql_initial_join = '''
            SELECT 
                a.loan_number,
                b.listing_number
            FROM (select 
                    loan_number,
                    amount_borrowed,
                    origination_date,
                    ifnull(prosper_rating, 'NA') AS prosper_rating,
                    borrower_rate
                FROM loan 
                --WHERE origination_date >= date('{year}-12-01') 
                --    AND origination_date <= date('{year}-12-31')
                ) a
            INNER JOIN (select 
                    listing_number,
                    amount_funded,
                    loan_origination_date,
                    ifnull(prosper_rating, 'NA') AS prosper_rating,
                    borrower_rate
                    
                FROM listing 
                ) b
            WHERE a.amount_borrowed = b.amount_funded
            AND a.origination_date = b.loan_origination_date
            AND a.prosper_rating = b.prosper_rating
            AND a.borrower_rate = b.borrower_rate
            
        
            '''

    output = pd.read_sql_query(sql_initial_join, connection)
    # Removes items that have duplicate matches
    mod = output[output.groupby('loan_number')['loan_number'].transform('count') == 1]
    mod = mod[mod.groupby('listing_number')['listing_number'].transform('count') == 1]

    mod.to_sql('join_reference', connection, if_exists='replace')


    # Creates database table with
    sql = '''
        SELECT 
            b.loan_number,
            b.amount_borrowed,
            b.term,
            b.age_in_months,
            b.origination_date,
            b.days_past_due,
            b.principal_balance,
            b.service_fees_paid,
            b.principal_paid,
            b.interest_paid,
            b.prosper_fees_paid,
            b.late_fees_paid,
            b.debt_sale_proceeds_received,
            b.loan_status,
            b.loan_status_description,
            b.loan_default_reason,
            b.loan_default_reason_description,
            b.next_payment_due_date,
            b.next_payment_due_amount,
            b.co_borrower_application,
            c.*
        FROM join_reference a
        LEFT JOIN loan b
        ON a.loan_number = b.loan_number
        LEFT JOIN listing c
        ON a.listing_number = c.listing_number
        '''

    output_data = pd.read_sql_query(sql, connection)
    output_data.to_sql('joined_data', connection, if_exists='replace')
    connection.execute('VACUUM;')

    if output_csv:
        output_data.to_csv('joined_data.csv')  # Creates CSV output (what was shared to team)

# ----------------------------------------------------------------
# Helper function to load DB table from CSV. Useful during development when not re-loading zipped files
# ----------------------------------------------------------------
def read_from_csv(filename = 'joined_data.csv'):
    '''
    :param filename: string, filename/path of csv file to use as "joined_data" in database
    :return: None
    '''
    try:
        connection = sqlite3.connect("loan_database")
    except sqlite3.Error as error:
        print('Error occurred - ', error)

    # must be in same format as joined_data output in create_db_tables
    output_data = pd.read_csv(filename, low_memory = False)
    output_data.to_sql('joined_data', connection, if_exists='replace')
    connection.execute('VACUUM;')


# ----------------------------------------------------------------
# Add Notch Data
# ----------------------------------------------------------------

def add_notch_data(notch_datafile = 'notches.csv'):
    '''
    :param notch_datafile: string, name of file with loan numbers and model output (notches) for each loan
    :return: None
    '''

    try:
        connection = sqlite3.connect("loan_database")
    except sqlite3.Error as error:
        print('Error occurred - ', error)

    notch_df = pd.read_csv(notch_datafile, low_memory=False, encoding="ISO-8859-1")
    notch_df.rename(columns={'0': 'loan_number', '1': 'notches'}, inplace=True)
    notch_df.to_sql('notch_table', connection, if_exists='replace')

    join_notch_sql = '''
        SELECT a.*, b.notches FROM joined_data a
        LEFT JOIN notch_table b
        ON a.loan_number = b.loan_number
    '''

    notch_data = pd.read_sql_query(join_notch_sql, connection)
    notch_data.drop(columns={'level_0'}, inplace=True)
    notch_data.to_sql('final_data', connection, if_exists='replace')
    connection.execute('VACUUM;')


# ----------------------------------------------------------------
# Create database for use in visualizations
# ----------------------------------------------------------------
def data_viz_output(current_listing_file = 'current_listing_data.csv'):
    '''
    :param: None
    :return: None
    '''

    try:
        connection = sqlite3.connect("loan_database")
        connection_trim = sqlite3.connect("loan_database_submission")
    except sqlite3.Error as error:
        print('Error occurred - ', error)

    # these tables are created for the submission
    outputData_for_submission_sql = '''
    SELECT
        scorex,
        TUFicoRange,
        loan_status_description,
        term,
        origination_date,
        prosper_rating,
        notches,
        listing_monthly_payment,
        income_range_description,
        interest_paid,
        principal_balance,
        amount_borrowed,
        borrower_rate
    FROM final_data
    '''

    listingData_for_submission_sql = '''
    SELECT
        listing_number,
        listing_amount,
        amount_funded,
        percent_funded,
        funding_threshold,
        prosper_rating,
        notches,
        TUFicoRange,
        scorex,
        listing_term
    FROM current_listing_table
    '''

    # mock active listings pulled from the provided CSV
    current_listing_table = pd.read_csv('current_listing_data.csv', low_memory=False, encoding="ISO-8859-1")
    current_listing_table.to_sql('current_listing_table', connection, if_exists='replace')

    # load these from full loan_database into database included with submission
    pd.read_sql_query(outputData_for_submission_sql, connection).to_sql('final_data', connection_trim, if_exists='replace')
    pd.read_sql_query(listingData_for_submission_sql, connection).to_sql('current_listing_table', connection_trim, if_exists='replace')
    connection_trim.execute('VACUUM;')
    connection.execute('VACUUM;')


def _main_(create_tables=False, load_from_csv=False, add_notches=False, create_data_viz_db=False):

    if create_tables:
        create_db_tables(output_csv=False)

    if load_from_csv:
        read_from_csv('joined_data.csv')

    if add_notches:
        add_notch_data('notches.csv')

    if create_data_viz_db:
        data_viz_output()


_main_(create_tables = create_db_tables_from_prosper,
       load_from_csv = load_joined_from_csv,
       add_notches = add_model_output,
       create_data_viz_db = create_data_viz_tables)
