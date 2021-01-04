from orm import drop_table, create_table, insert_text, show_all_texts_in_db, update_ckecked_status, \
    update_sent_to_client_status, Texts, delete_texts_which_are_sent_to_client




def main():

    create_table()

    insert_text("first")
    insert_text("second")
    insert_text("third")
    insert_text("forth")
    show_all_texts_in_db()
    print("--------------------------")

    update_ckecked_status(hash=Texts.get(Texts.text == "first").id)
    update_sent_to_client_status(hash=Texts.get(Texts.text == "first").id)
    show_all_texts_in_db()
    print("--------------------------")

    delete_texts_which_are_sent_to_client()
    show_all_texts_in_db()
    print("--------------------------")


if __name__ == "__main__":
    main()